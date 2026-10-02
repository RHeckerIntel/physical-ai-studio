"""Run one robot at its own rate against the session's feature store.

The worker reads its hardware and publishes the observation, then -- if it has
been told to drive the robot -- takes the action features out of the store and
writes them. It does not know or care who authored those actions, which is what
makes the sources interchangeable: a leader arm, a keyboard, a policy, or a
start protocol ramping to a home position are all just something that writes
action features. Teleoperation is a key mapping rather than a mode.

Partial authorship follows from the same thing. Two sources writing disjoint
joints -- a keyboard driving a base while a leader drives the arm -- need no
coordination, because the worker only ever reads the store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from runtimev2.features import ACTION_PREFIX, OBSERVATION_PREFIX, joint_feature_key

if TYPE_CHECKING:
    from physicalai.robot.interface import Robot

    from runtimev2.store import FeatureStore


class RobotWorker:
    """Publish one robot's observation, and optionally drive it from the store.

    ``tick`` is a single pass and does its own timing-free work, so a caller --
    or a test -- drives it as fast or as slowly as it likes. The rate loop is
    the caller's concern.
    """

    def __init__(
        self,
        robot: Robot,
        store: FeatureStore,
        *,
        name: str,
        write_actions: bool = False,
    ) -> None:
        self._robot = robot
        self._store = store
        self._name = name
        self._write_actions = write_actions
        # Resolved once: the key order is also the order ``send_action`` expects,
        # so it must track ``joint_names`` rather than the store's sorting.
        self._joint_names = list(robot.joint_names)
        self._observation_keys = [joint_feature_key(OBSERVATION_PREFIX, name, joint) for joint in self._joint_names]
        self._action_keys = [joint_feature_key(ACTION_PREFIX, name, joint) for joint in self._joint_names]

    @property
    def name(self) -> str:
        return self._name

    @property
    def observation_keys(self) -> tuple[str, ...]:
        return tuple(self._observation_keys)

    @property
    def action_keys(self) -> tuple[str, ...]:
        return tuple(self._action_keys)

    @property
    def write_actions(self) -> bool:
        """Whether this worker drives its robot from the store's action features."""
        return self._write_actions

    @write_actions.setter
    def write_actions(self, enabled: bool) -> None:
        self._write_actions = enabled

    def tick(self) -> None:
        """Publish the observation, then drive the robot if enabled."""
        self._publish_observation()
        if self._write_actions:
            self._drive()

    def _publish_observation(self) -> None:
        observation = self._robot.get_observation()
        values = {
            key: float(position)
            for key, position in zip(self._observation_keys, observation.joint_positions, strict=True)
        }
        # The robot's own capture time, so a reader sees when the measurement
        # was taken rather than when it was stored.
        self._store.write_many(values, timestamp=observation.timestamp)

    def _drive(self) -> None:
        """Send the store's action features to the robot.

        Waits until every joint has been authored at least once. A partially
        written action vector has no safe filler -- holding a position the
        robot was never commanded to is as arbitrary as sending a zero -- so
        the worker stays passive until some source has spoken for each joint.
        """
        samples = self._store.snapshot(self._action_keys)
        if len(samples) != len(self._action_keys):
            return
        action = np.array([samples[key].value for key in self._action_keys], dtype=np.float32)
        try:
            self._robot.send_action(action)
        except Exception:
            logger.exception("Robot {} rejected an action", self._name)
            raise

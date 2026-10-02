"""Own one robot: publish what it measures, and drive it from the store.

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

from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from runtimev2.features import ACTION_PREFIX, OBSERVATION_PREFIX, joint_feature_key
from runtimev2.workers.base import ThreadedWorker

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicalai.robot.interface import Robot

    from runtimev2.environment import RobotShape
    from runtimev2.store import FeatureStore


class RobotWorker(ThreadedWorker):
    """Hold one robot open, publish its observation, and optionally drive it.

    Takes the robot directly rather than a builder: unlike a camera, a robot
    that will not connect is not going to on a second attempt -- it is a port
    that is wrong or a device that is gone.

    The key order comes from the shape, not the connected robot, because that is
    the order the feature spec and therefore the action vector use. The two are
    checked against each other on acquire.
    """

    def __init__(
        self,
        robot: Robot,
        store: FeatureStore,
        *,
        shape: RobotShape,
        hz: float,
        write_actions: bool = False,
    ) -> None:
        super().__init__(name=shape.key, hz=hz)
        self._robot = robot
        self._store = store
        self._shape = shape
        self._write_actions = write_actions
        self._joint_names = list(shape.joint_names)
        self._observation_keys = [joint_feature_key(OBSERVATION_PREFIX, shape.key, j) for j in self._joint_names]
        self._action_keys = [joint_feature_key(ACTION_PREFIX, shape.key, j) for j in self._joint_names]
        self._connected = False

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

    @contextmanager
    def acquire(self) -> Generator[None]:
        """Connect the robot, and disconnect it after.

        Runs on this worker's thread, so spawning the owner process that holds
        the hardware and waiting for it to let go both block here rather than on
        the event loop.
        """
        self._robot.connect()
        self._connected = True
        try:
            self._verify_joint_order()
            yield
        finally:
            self._connected = False
            self._robot.disconnect()

    def _verify_joint_order(self) -> None:
        """Fail if the connected robot disagrees with the shape it was described by.

        The spec's joint order comes from a locally constructed driver; the
        device's comes from the owner process's metadata. An action vector is
        assembled in the spec's order, so a disagreement would send joint
        values to the wrong joints -- silently, and while the arm is live.

        Raises:
            RuntimeError: The orders differ.
        """
        connected = tuple(self._robot.joint_names)
        if connected != self._shape.joint_names:
            raise RuntimeError(
                f"Robot {self._shape.key} reports joints {connected} once connected, "
                f"but was described as {self._shape.joint_names}"
            )

    def tick(self) -> None:
        """Publish the observation, then drive the robot if enabled.

        Raises:
            RuntimeError: Ticked outside ``acquire``.
        """
        if not self._connected:
            raise RuntimeError(f"Robot {self._shape.key} is not connected")
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
            logger.exception("Robot {} rejected an action", self._shape.key)
            raise

"""Own one robot: publish what it measures, and follow its action features.

The worker publishes its observation and sends whatever the action features
say, at its own rate, always. It does not know what wrote them -- a leader arm,
a policy, a protocol ramping to a home position all look the same from here.

On connecting it writes its own measured position into those features, so the
first thing it commands is where the arm already is. That is what makes loading
safe without a switch: there is nothing to enable, because following an action
equal to the current position moves nothing. Whether the arm then moves is
entirely up to whether something writes a different one, which is the control's
job -- so no control means no motion, with no second state to get wrong.

Because the writer runs at its own rate, consecutive actions arrive spaced by
*its* period, not this worker's. Sending each one as a step would move the arm
in jumps as coarse as that period, so the worker interpolates: it ramps from
the previous action to the newest across the gap between their timestamps, at
its own much finer rate. The timestamps make that stable even though the two
loops are unsynchronised -- all the worker needs is the spacing, not a shared
clock.

It clamps at the newest action and never extrapolates past it. A control that
stalls therefore leaves the arm holding its last command, rather than being
carried somewhere no one asked for.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from runtimev2.features import ACTION_PREFIX, OBSERVATION_PREFIX, joint_feature_key
from runtimev2.workers.base import ThreadedWorker

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from physicalai.robot.interface import Robot

    from runtimev2.environment import RobotShape
    from runtimev2.session_store import SessionStore


MAX_GOAL_TIME_S = 0.5
"""Longest ramp to a newly written action. See ``_begin_segment``."""


class RobotWorker(ThreadedWorker):
    """Hold one robot open, publish its observation, and optionally drive it.

    Takes the robot directly rather than a builder: unlike a camera, one that
    will not connect is a wrong port or a missing device, not bad luck.

    Key order comes from the shape, since that is the order the feature spec and
    the action vector use; the connected robot is checked against it on acquire.

    With nothing writing its action features it holds the position it had when
    it connected, which is what a follower does until a control is selected.
    """

    def __init__(
        self,
        robot: Robot,
        store: SessionStore,
        *,
        shape: RobotShape,
        hz: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """``clock`` times the ramp; it is a parameter so a test can step it."""
        super().__init__(name=shape.key, hz=hz)
        self._clock = clock
        self._robot = robot
        self._store = store
        self._shape = shape
        self._joint_names = list(shape.joint_names)
        # The segment currently being ramped across, and when this worker first
        # saw its target. Timed from arrival rather than from the writer's
        # clock, which is not this one.
        self._from: np.ndarray | None = None
        self._to: np.ndarray | None = None
        self._goal_time = 0.0
        self._target_timestamp: float | None = None
        self._target_seen_at = 0.0
        self._observation_keys = [joint_feature_key(OBSERVATION_PREFIX, shape.key, j) for j in self._joint_names]
        self._action_keys = [joint_feature_key(ACTION_PREFIX, shape.key, j) for j in self._joint_names]
        self._connected = False

    @property
    def observation_keys(self) -> tuple[str, ...]:
        return tuple(self._observation_keys)

    @property
    def action_keys(self) -> tuple[str, ...]:
        return tuple(self._action_keys)

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
            self._hold_current_position()
            yield
        finally:
            self._connected = False
            self._robot.disconnect()

    def _verify_joint_order(self) -> None:
        """Fail if the connected robot disagrees with the shape describing it.

        The shape and this driver come from separate builds, each resolving a
        live port, so a device swapped on that path arrives under the other
        one's shape. The action vector follows the shape's order, so the
        mismatch would drive joint values into the wrong joints while live.

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
        """Publish the observation, then follow the action features.

        Raises:
            RuntimeError: Ticked outside ``acquire``.
        """
        if not self._connected:
            raise RuntimeError(f"Robot {self._shape.key} is not connected")
        _t0 = time.monotonic()
        observation = self._robot.get_observation()
        _t1 = time.monotonic()
        positions = np.asarray(observation.joint_positions, dtype=np.float32)
        self._store.write_many(
            {key: float(value) for key, value in zip(self._observation_keys, positions, strict=True)},
            timestamp=observation.timestamp,
        )
        self._drive(positions)
        if time.monotonic() - _t0 > 0.03:
            logger.warning("SLOWTICK {} read={:.1f}ms", self._shape.key, (_t1 - _t0) * 1000)

    def _hold_current_position(self) -> None:
        """Seed the action features with where the arm is.

        So the first command is a command to stay put. Without this the worker
        would either need a switch or would follow whatever a previous
        environment left in a store -- and a freshly loaded environment has no
        action to follow at all.
        """
        observation = self._robot.get_observation()
        positions = np.asarray(observation.joint_positions, dtype=np.float32)
        self._store.write_many(
            {key: float(value) for key, value in zip(self._action_keys, positions, strict=True)},
            timestamp=observation.timestamp,
        )

    def _publish_observation(self) -> np.ndarray:
        """Publish the robot's current joints, and return them."""
        observation = self._robot.get_observation()
        positions = np.asarray(observation.joint_positions, dtype=np.float32)
        values = {key: float(position) for key, position in zip(self._observation_keys, positions, strict=True)}
        # The robot's own capture time, so a reader sees when the measurement
        # was taken rather than when it was stored.
        self._store.write_many(values, timestamp=observation.timestamp)
        return positions

    def _drive(self, positions: np.ndarray) -> None:
        """Send where the arm should be now, ramping toward the newest action.

        Waits until every joint has been written. A partly written action has no
        safe filler -- holding a position the robot was never commanded to is as
        arbitrary as sending a zero -- so nothing is sent until some control has
        spoken for each joint.
        """
        samples = self._store.snapshot(self._action_keys)
        if len(samples) != len(self._action_keys):
            return
        target = np.array([samples[key].value for key in self._action_keys], dtype=np.float32)
        timestamp = max(sample.timestamp for sample in samples.values())
        if timestamp != self._target_timestamp:
            self._begin_segment(positions, target, timestamp)
        command = self._interpolate(target)
        try:
            self._robot.send_action(command)
        except Exception:
            logger.exception("Robot {} rejected an action", self._shape.key)
            raise

    def _begin_segment(self, positions: np.ndarray, target: np.ndarray, timestamp: float) -> None:
        """Start ramping toward a newly written action.

        The ramp starts from the previous target rather than from the measured
        position, so a follower lagging behind is not treated as the place the
        trajectory came from -- which would make it lag further every segment.
        The measured position is only the starting point for the first one.
        """
        previous = self._target_timestamp
        self._from = self._to if self._to is not None else positions
        self._to = target
        # The writer's own period, measured rather than configured, so a
        # control that changes rate is followed without being told. Capped
        # because a long gap does not mean a slow control: it means a new one
        # started writing, and the arm should approach its first command
        # briskly rather than creeping there over however long the pause was.
        gap = max(timestamp - previous, 0.0) if previous is not None else 0.0
        self._goal_time = min(gap, MAX_GOAL_TIME_S)
        self._target_timestamp = timestamp
        self._target_seen_at = self._clock()

    def _interpolate(self, target: np.ndarray) -> np.ndarray:
        """Where along the current segment the arm should be now.

        Clamped at the target: a control that stalls leaves the arm holding its
        last command rather than being carried past it.
        """
        if self._from is None or self._goal_time <= 0.0:
            return target
        fraction = (self._clock() - self._target_seen_at) / self._goal_time
        if fraction >= 1.0:
            return target
        return self._from + (target - self._from) * np.float32(fraction)

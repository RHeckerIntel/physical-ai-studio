"""A robot worker publishes what it measures and commands what its control says."""

from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pytest
from loguru import logger

from runtimev2.environment import RobotShape
from runtimev2.features import ACTION_PREFIX, OBSERVATION_PREFIX, FeatureSpec, joint_feature_key, robot_features
from runtimev2.store import FeatureStore
from runtimev2.workers.loop import run_at
from runtimev2.workers.robot import RobotWorker

if TYPE_CHECKING:
    from collections.abc import Iterator

    from physicalai.capture.frame import Frame

JOINTS = ["shoulder_pan", "elbow_flex", "gripper"]


@dataclass
class _Observation:
    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict[str, Frame] | None = None

    @property
    def state(self) -> np.ndarray:
        return self.joint_positions


@dataclass
class _FakeRobot:
    """Stands in for a physicalai ``Robot`` without any hardware."""

    joint_names: list[str] = field(default_factory=lambda: list(JOINTS))
    positions: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.2, 0.3], dtype=np.float32))
    timestamp: float = 100.0
    sent: list[np.ndarray] = field(default_factory=list)
    tick_delay: float = 0.0

    connected: bool = False
    disconnects: int = 0

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False
        self.disconnects += 1

    def get_observation(self) -> _Observation:
        if self.tick_delay:
            time.sleep(self.tick_delay)
        return _Observation(joint_positions=self.positions, timestamp=self.timestamp)

    def send_action(self, action: np.ndarray) -> None:
        self.sent.append(np.array(action))


SHAPE = RobotShape(key="follower", robot_id="r0", role="follower", joint_names=tuple(JOINTS))


class _Clock:
    """A clock a test steps by hand, so a ramp is not timing-dependent."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _setup(*, clock: _Clock | None = None, seed: bool = True) -> tuple[_FakeRobot, FeatureStore, RobotWorker]:
    """A connected worker, seeded as ``acquire`` would unless ``seed`` is off.

    These tests drive ticks directly rather than through the worker's thread.
    """
    robot = _FakeRobot()
    store = FeatureStore(FeatureSpec.build(robot_features("follower", JOINTS)))
    worker = RobotWorker(robot, store, shape=SHAPE, hz=100.0, clock=clock or _Clock())
    robot.connect()
    worker._connected = True
    if seed:
        worker._hold_current_position()
    return robot, store, worker


def _action_key(joint: str) -> str:
    return joint_feature_key(ACTION_PREFIX, "follower", joint)


def _write_action(store: FeatureStore, values: list[float], timestamp: float) -> None:
    """Write what a control would, without running one."""
    store.write_many(
        {_action_key(joint): value for joint, value in zip(JOINTS, values, strict=True)},
        timestamp=timestamp,
    )


class TestObservation:
    def test_a_tick_publishes_every_joint(self) -> None:
        _robot, store, worker = _setup()

        worker.tick()

        shot = store.snapshot(worker.observation_keys)
        assert len(shot) == len(JOINTS)
        assert shot[joint_feature_key(OBSERVATION_PREFIX, "follower", "shoulder_pan")].value == pytest.approx(0.1)

    def test_the_robots_own_capture_time_is_kept(self) -> None:
        """Storing arrival time instead would fold transport delay into the reading."""
        robot, store, worker = _setup()
        robot.timestamp = 1234.5

        worker.tick()

        sample = store.read(joint_feature_key(OBSERVATION_PREFIX, "follower", "gripper"))
        assert sample is not None
        assert sample.timestamp == 1234.5


class TestHoldingOnConnect:
    """An arm follows its action features always, so they start where it is."""

    def test_connecting_seeds_the_action_with_the_measured_position(self) -> None:
        robot, store, _worker = _setup()

        for joint, position in zip(JOINTS, robot.positions, strict=True):
            assert store.read(_action_key(joint)).value == pytest.approx(float(position))

    def test_the_first_command_is_where_the_arm_already_is(self) -> None:
        """Which is what makes loading safe with no switch to forget."""
        robot, _store, worker = _setup()

        worker.tick()

        np.testing.assert_allclose(robot.sent[0], robot.positions)

    def test_it_keeps_holding_while_nothing_writes(self) -> None:
        robot, _store, worker = _setup()

        for _ in range(5):
            worker.tick()

        assert len(robot.sent) == 5
        for sent in robot.sent:
            np.testing.assert_allclose(sent, robot.positions)

    def test_nothing_is_sent_before_an_action_exists(self) -> None:
        """A worker ticked without acquiring has nothing to follow."""
        robot, _store, worker = _setup(seed=False)

        worker.tick()

        assert robot.sent == []

    def test_a_partially_written_action_is_not_sent(self) -> None:
        """There is no safe filler for a joint nothing has spoken for."""
        robot, store, worker = _setup(seed=False)
        store.write(_action_key("shoulder_pan"), 1.0, timestamp=1.0)

        worker.tick()

        assert robot.sent == []


class TestDriving:
    def test_the_action_is_sent_in_joint_order(self) -> None:
        """``send_action`` takes a vector, so the order must track joint_names."""
        robot, store, worker = _setup()
        _write_action(store, [1.0, 2.0, 3.0], 1000.0)

        worker.tick()

        np.testing.assert_allclose(robot.sent[-1], [1.0, 2.0, 3.0])

    def test_the_observation_is_still_published_while_driving(self) -> None:
        _robot, store, worker = _setup()
        _write_action(store, [1.0, 2.0, 3.0], 1000.0)

        worker.tick()

        assert len(store.snapshot(worker.observation_keys)) == len(JOINTS)


class TestRamping:
    """A control writes at its own rate; the arm is moved across the gap.

    Sending each action as a step would move the arm in jumps as coarse as the
    writer's period. The timestamps give the spacing without the two loops
    sharing a clock.
    """

    def test_a_later_action_is_approached_across_its_own_period(self) -> None:
        clock = _Clock()
        robot, store, worker = _setup(clock=clock)
        _write_action(store, [0.0, 0.0, 0.0], 1.0)
        worker.tick()

        # Written 100ms after the last, so the ramp spans 100ms.
        _write_action(store, [10.0, 10.0, 10.0], 1.1)
        worker.tick()
        clock.advance(0.05)
        worker.tick()

        np.testing.assert_allclose(robot.sent[-1], [5.0, 5.0, 5.0], atol=1e-5)

    def test_it_arrives_at_the_target(self) -> None:
        clock = _Clock()
        robot, store, worker = _setup(clock=clock)
        _write_action(store, [0.0, 0.0, 0.0], 1.0)
        worker.tick()
        _write_action(store, [10.0, 10.0, 10.0], 1.1)
        worker.tick()

        clock.advance(0.1)
        worker.tick()

        np.testing.assert_allclose(robot.sent[-1], [10.0, 10.0, 10.0])

    def test_it_never_goes_past_the_target(self) -> None:
        """A control that stalls leaves the arm holding its last command."""
        clock = _Clock()
        robot, store, worker = _setup(clock=clock)
        _write_action(store, [0.0, 0.0, 0.0], 1.0)
        worker.tick()
        _write_action(store, [10.0, 10.0, 10.0], 1.1)
        worker.tick()

        clock.advance(5.0)
        worker.tick()
        worker.tick()

        np.testing.assert_allclose(robot.sent[-1], [10.0, 10.0, 10.0])

    def test_a_slow_control_is_still_commanded_every_tick(self) -> None:
        """The point of ramping: many small commands, not one step per action."""
        clock = _Clock()
        robot, store, worker = _setup(clock=clock)
        _write_action(store, [0.0, 0.0, 0.0], 1.0)
        worker.tick()
        _write_action(store, [10.0, 10.0, 10.0], 1.2)

        for _ in range(5):
            worker.tick()
            clock.advance(0.01)

        assert len(robot.sent) == 6
        moved = [float(sent[0]) for sent in robot.sent[1:]]
        assert moved == sorted(moved), "the ramp did not advance monotonically"
        assert 0.0 < moved[-1] < 10.0, f"expected a partial ramp, got {moved[-1]}"

    def test_a_ramp_starts_from_the_previous_target_not_the_measurement(self) -> None:
        """A follower lagging behind must not make the trajectory lag further."""
        clock = _Clock()
        robot, store, worker = _setup(clock=clock)
        _write_action(store, [10.0, 10.0, 10.0], 1.0)
        worker.tick()
        # The arm is nowhere near its command; the next ramp starts from the
        # command, not from here.
        robot.positions = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        _write_action(store, [20.0, 20.0, 20.0], 1.1)
        worker.tick()
        clock.advance(0.05)
        worker.tick()

        np.testing.assert_allclose(robot.sent[-1], [15.0, 15.0, 15.0], atol=1e-5)


@contextmanager
def _captured_warnings() -> Iterator[list[str]]:
    """Collect loguru warnings; caplog only sees stdlib logging."""
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


class TestRunAt:
    """Whether the rate is being met is the loop's business, not the store's."""

    def test_it_ticks_until_told_to_stop(self) -> None:
        robot, _store, worker = _setup()
        ticks = 0

        def should_stop() -> bool:
            nonlocal ticks
            ticks += 1
            return ticks > 3

        run_at(worker, hz=1000, should_stop=should_stop)

        assert ticks == 4
        assert len(robot.sent) == 3, "the loop checked four times, so it ticked three"

    def test_a_slow_tick_is_reported_once_per_interval(self) -> None:
        """At 100Hz a line per overrun would bury the signal in its own noise."""
        robot, _store, worker = _setup()
        robot.tick_delay = 0.02  # 20ms against a 1ms period
        calls = 0

        def should_stop() -> bool:
            nonlocal calls
            calls += 1
            return calls > 60  # ~1.2s of overruns, so one report is due

        with _captured_warnings() as messages:
            run_at(worker, hz=1000, should_stop=should_stop)

        reports = [message for message in messages if "missed" in message]
        assert len(reports) == 1, f"expected one throttled report, got {len(reports)}"
        assert "follower" in reports[0]

    def test_a_loop_that_keeps_up_reports_nothing(self) -> None:
        _robot, _store, worker = _setup()
        calls = 0

        def should_stop() -> bool:
            nonlocal calls
            calls += 1
            return calls > 3

        with _captured_warnings() as messages:
            run_at(worker, hz=50, should_stop=should_stop)

        assert [message for message in messages if "missed" in message] == []


class TestAsAWorker:
    """The worker owns its robot and its thread, not just a tick."""

    async def test_entering_connects_and_leaving_disconnects(self) -> None:
        robot = _FakeRobot()
        store = FeatureStore(FeatureSpec.build(robot_features("follower", JOINTS)))
        worker = RobotWorker(robot, store, shape=SHAPE, hz=100.0)

        async with worker:
            assert robot.connected
            await asyncio.sleep(0.1)
            assert len(store.snapshot(worker.observation_keys)) == len(JOINTS), "the worker's own thread never ticked"

        assert not robot.connected
        assert robot.disconnects == 1

    async def test_a_robot_that_disagrees_on_joint_order_is_refused(self) -> None:
        """The action vector is built in the described order, so a mismatch would
        send joint values to the wrong joints while the arm is live."""
        robot = _FakeRobot()
        robot.joint_names = list(reversed(JOINTS))
        store = FeatureStore(FeatureSpec.build(robot_features("follower", JOINTS)))
        worker = RobotWorker(robot, store, shape=SHAPE, hz=100.0)

        with pytest.raises(RuntimeError, match="once connected"):
            async with worker:
                pass

        assert robot.disconnects == 1, "a refused robot was left connected"

    async def test_ticking_outside_acquire_is_refused(self) -> None:
        robot = _FakeRobot()
        store = FeatureStore(FeatureSpec.build(robot_features("follower", JOINTS)))
        worker = RobotWorker(robot, store, shape=SHAPE, hz=100.0)

        with pytest.raises(RuntimeError, match="not connected"):
            worker.tick()

"""A robot worker publishes what it measures and drives what the store tells it to."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pytest
from loguru import logger

from runtimev2.features import ACTION_PREFIX, OBSERVATION_PREFIX, FeatureSpec, joint_feature_key, robot_features
from runtimev2.store import FeatureStore
from runtimev2.workers.robot import RobotWorker, run_at

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

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...

    def get_observation(self) -> _Observation:
        if self.tick_delay:
            time.sleep(self.tick_delay)
        return _Observation(joint_positions=self.positions, timestamp=self.timestamp)

    def send_action(self, action: np.ndarray) -> None:
        self.sent.append(np.array(action))


def _setup(*, write_actions: bool = False) -> tuple[_FakeRobot, FeatureStore, RobotWorker]:
    robot = _FakeRobot()
    store = FeatureStore(FeatureSpec.build(robot_features("follower", JOINTS)))
    worker = RobotWorker(robot, store, name="follower", write_actions=write_actions)
    return robot, store, worker


def _action_key(joint: str) -> str:
    return joint_feature_key(ACTION_PREFIX, "follower", joint)


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


class TestDriving:
    def test_nothing_is_sent_when_writing_is_disabled(self) -> None:
        robot, store, worker = _setup(write_actions=False)
        for joint in JOINTS:
            store.write(_action_key(joint), 1.0, timestamp=1.0)

        worker.tick()

        assert robot.sent == []

    def test_an_authored_action_is_sent_in_joint_order(self) -> None:
        """``send_action`` takes a vector, so the order must track joint_names, not the store's sorting."""
        robot, store, worker = _setup(write_actions=True)
        for value, joint in enumerate(JOINTS, start=1):
            store.write(_action_key(joint), float(value), timestamp=1.0)

        worker.tick()

        assert len(robot.sent) == 1
        np.testing.assert_allclose(robot.sent[0], [1.0, 2.0, 3.0])

    def test_a_partially_authored_action_is_not_sent(self) -> None:
        """There is no safe filler for an unauthored joint; holding still is the caller's job."""
        robot, store, worker = _setup(write_actions=True)
        store.write(_action_key("shoulder_pan"), 1.0, timestamp=1.0)

        worker.tick()

        assert robot.sent == []

    def test_disjoint_sources_compose_without_coordination(self) -> None:
        """A keyboard driving one joint and a leader driving the rest need not know about each other."""
        robot, store, worker = _setup(write_actions=True)
        store.write(_action_key("gripper"), 9.0, timestamp=1.0)  # "keyboard"
        for joint in ("shoulder_pan", "elbow_flex"):  # "leader"
            store.write(_action_key(joint), 5.0, timestamp=1.0)

        worker.tick()

        np.testing.assert_allclose(robot.sent[0], [5.0, 5.0, 9.0])

    def test_writing_can_be_turned_on_mid_session(self) -> None:
        robot, store, worker = _setup(write_actions=False)
        for joint in JOINTS:
            store.write(_action_key(joint), 1.0, timestamp=1.0)

        worker.tick()
        worker.write_actions = True
        worker.tick()

        assert len(robot.sent) == 1

    def test_the_observation_is_still_published_while_driving(self) -> None:
        _robot, store, worker = _setup(write_actions=True)

        worker.tick()

        assert len(store.snapshot(worker.observation_keys)) == len(JOINTS)


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

        assert robot.sent == []
        assert ticks == 4

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

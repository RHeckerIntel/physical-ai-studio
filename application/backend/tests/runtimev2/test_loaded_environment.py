"""A loaded environment holds the devices, the store and the workers."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

from runtimev2.features import ACTION_PREFIX, OBSERVATION_PREFIX, joint_feature_key
from runtimev2.loaded_environment import LoadedEnvironment

JOINTS = ("shoulder_pan", "gripper")


@dataclass
class _Observation:
    joint_positions: np.ndarray
    timestamp: float
    sensor_data: None = None
    images: None = None

    @property
    def state(self) -> np.ndarray:
        return self.joint_positions


@dataclass
class _SharedRobot:
    """Stands in for a connected SharedRobot, including its connect-gated joints."""

    label: str
    reported_joints: tuple[str, ...] = JOINTS
    position: float = 0.0
    connected: bool = False
    disconnects: int = 0
    sent: list[np.ndarray] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def joint_names(self) -> list[str]:
        if not self.connected:
            raise RuntimeError("SharedRobot is not connected. Call connect() first.")
        return list(self.reported_joints)

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False
        self.disconnects += 1

    def get_observation(self) -> _Observation:
        with self.lock:
            return _Observation(
                joint_positions=np.full(len(JOINTS), self.position, dtype=np.float32),
                timestamp=100.0 + self.position,
            )

    def send_action(self, action: np.ndarray, **_: Any) -> None:
        with self.lock:
            self.sent.append(np.array(action))


@dataclass
class _Row:
    name: str
    type: str = "SO101"
    id: Any = field(default_factory=uuid4)
    connection_string: str = "/dev/ttyACM0"


@dataclass
class _Teleoperator:
    robot: _Row | None = None


@dataclass
class _Configured:
    robot: _Row
    tele_operator: _Teleoperator = field(default_factory=_Teleoperator)


@dataclass
class _Environment:
    name: str = "test env"
    robots: list[_Configured] = field(default_factory=list)
    cameras: list[Any] = field(default_factory=list)


@dataclass
class _Definition:
    role: str


@dataclass
class _Factory:
    roles: dict[str, str] = field(default_factory=dict)
    joints: dict[str, tuple[str, ...]] = field(default_factory=dict)
    robots: dict[str, _SharedRobot] = field(default_factory=dict)

    async def build_robot_driver(self, robot: _Row, port_finder: object) -> tuple[Any, _Definition]:
        driver = _SharedRobot(robot.name, connected=True)  # a local driver needs no connection
        driver.reported_joints = self.joints.get(robot.name, JOINTS)
        return driver, _Definition(role=self.roles.get(robot.name, "follower"))

    async def build_shared_robot(self, robot: _Row) -> tuple[_SharedRobot, _Definition]:
        shared = _SharedRobot(robot.name, reported_joints=self.joints.get(robot.name, JOINTS))
        self.robots[robot.name] = shared
        return shared, _Definition(role=self.roles.get(robot.name, "follower"))

    async def find_port(self, port_info: object) -> str | None:
        return None


def _teleop_env() -> tuple[_Environment, _Factory]:
    environment = _Environment(
        robots=[_Configured(robot=_Row("follower"), tele_operator=_Teleoperator(robot=_Row("leader")))]
    )
    return environment, _Factory(roles={"follower": "follower", "leader": "leader"})


async def _settle() -> None:
    """Give the worker threads a few ticks to run."""
    await asyncio.sleep(0.2)


class TestLifecycle:
    async def test_opening_connects_every_robot_and_closing_releases_them(self) -> None:
        environment, factory = _teleop_env()

        async with LoadedEnvironment(environment, factory) as loaded:
            assert set(loaded.state().robots) == {"follower", "leader"}
            assert all(robot.connected for robot in factory.robots.values())

        assert all(not robot.connected for robot in factory.robots.values())
        assert all(robot.disconnects == 1 for robot in factory.robots.values())

    async def test_observations_reach_the_store_on_their_own(self) -> None:
        """No one drives the session; the robot workers tick themselves."""
        environment, factory = _teleop_env()

        async with LoadedEnvironment(environment, factory) as loaded:
            factory.robots["leader"].position = 2.0
            await _settle()
            sample = loaded.store.read(joint_feature_key(OBSERVATION_PREFIX, "leader", "gripper"))

        assert sample is not None
        assert sample.value == pytest.approx(2.0)

    async def test_a_disagreeing_robot_is_refused(self) -> None:
        """The action vector is built in the described order; a mismatch would
        send joint values to the wrong joints while the arm is live."""
        environment, factory = _teleop_env()
        factory.joints = {"follower": JOINTS, "leader": JOINTS}
        original = factory.build_shared_robot

        async def disagree(robot: _Row) -> tuple[_SharedRobot, _Definition]:
            shared, definition = await original(robot)
            shared.reported_joints = ("gripper", "shoulder_pan")  # reversed
            return shared, definition

        factory.build_shared_robot = disagree  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="once connected"):
            async with LoadedEnvironment(environment, factory):
                pass

    async def test_a_robot_connected_before_the_failure_is_still_released(self) -> None:
        environment, factory = _teleop_env()
        original = factory.build_shared_robot
        calls = 0

        async def fail_on_second(robot: _Row) -> tuple[_SharedRobot, _Definition]:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("owner process refused")
            return await original(robot)

        factory.build_shared_robot = fail_on_second  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="owner process refused"):
            async with LoadedEnvironment(environment, factory):
                pass

        assert [robot.disconnects for robot in factory.robots.values()] == [1]


class TestTeleoperation:
    async def test_nothing_is_driven_until_it_is_enabled(self) -> None:
        environment, factory = _teleop_env()

        async with LoadedEnvironment(environment, factory):
            factory.robots["leader"].position = 1.0
            await _settle()

        assert factory.robots["follower"].sent == []

    async def test_enabling_sends_the_leaders_position_to_the_follower(self) -> None:
        environment, factory = _teleop_env()

        async with LoadedEnvironment(environment, factory) as loaded:
            factory.robots["leader"].position = 1.25
            await _settle()  # let the mapping publish before writes are enabled
            loaded.set_teleoperating(True)
            await _settle()
            assert loaded.state().teleoperating

        sent = factory.robots["follower"].sent
        assert sent, "the follower was never commanded"
        np.testing.assert_allclose(sent[-1], [1.25, 1.25])

    async def test_the_mapping_publishes_even_while_disabled(self) -> None:
        """A client can watch what would be commanded before committing to it."""
        environment, factory = _teleop_env()

        async with LoadedEnvironment(environment, factory) as loaded:
            factory.robots["leader"].position = 3.0
            await _settle()
            action = loaded.store.read(joint_feature_key(ACTION_PREFIX, "follower", "gripper"))

        assert action is not None
        assert action.value == pytest.approx(3.0)
        assert factory.robots["follower"].sent == []

    async def test_disabling_stops_the_follower(self) -> None:
        environment, factory = _teleop_env()

        async with LoadedEnvironment(environment, factory) as loaded:
            factory.robots["leader"].position = 1.0
            loaded.set_teleoperating(True)
            await _settle()
            loaded.set_teleoperating(False)
            sent_when_disabled = len(factory.robots["follower"].sent)
            await _settle()

            assert len(factory.robots["follower"].sent) == sent_when_disabled
            assert not loaded.state().teleoperating

    async def test_an_environment_with_no_leader_has_nothing_to_teleoperate(self) -> None:
        environment = _Environment(robots=[_Configured(robot=_Row("follower"))])

        async with LoadedEnvironment(environment, _Factory()) as loaded:
            with pytest.raises(RuntimeError, match="no leader"):
                loaded.set_teleoperating(True)

    async def test_a_leader_with_a_different_joint_count_is_refused(self) -> None:
        environment, factory = _teleop_env()
        factory.joints = {"follower": JOINTS, "leader": ("only_one",)}

        with pytest.raises(ValueError, match="same number of joints"):
            async with LoadedEnvironment(environment, factory):
                pass

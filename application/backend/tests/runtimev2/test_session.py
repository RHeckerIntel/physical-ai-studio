"""The session holds at most one environment, and can swap it."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

from runtimev2.session import RuntimeSession

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
                joint_positions=np.full(len(self.reported_joints), self.position, dtype=np.float32),
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


class TestLoading:
    async def test_a_session_starts_empty(self) -> None:
        """Nothing is connected until an environment is asked for."""
        async with RuntimeSession(_Factory()) as session:
            assert session.environment is None
            assert session.state().loaded is False

    async def test_loading_connects_the_environments_robots(self) -> None:
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            loaded = await session.load(environment)

            assert session.environment is loaded
            assert session.state().loaded is True
            assert all(robot.connected for robot in factory.robots.values())

    async def test_unloading_releases_them_and_leaves_the_session_open(self) -> None:
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)
            await session.unload()

            assert session.environment is None
            assert all(not robot.connected for robot in factory.robots.values())
            # Still usable: the point of unloading is to load something else.
            await session.load(environment)
            assert session.environment is not None

    async def test_unloading_is_idempotent(self) -> None:
        async with RuntimeSession(_Factory()) as session:
            await session.unload()
            await session.unload()

            assert session.environment is None

    async def test_leaving_the_session_unloads_what_is_still_loaded(self) -> None:
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)

        assert all(not robot.connected for robot in factory.robots.values())

    async def test_loading_again_releases_the_previous_robots_first(self) -> None:
        """Two environments can share a robot, so the old one has to let go
        before the new one claims anything."""
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)
            first = dict(factory.robots)
            await session.load(environment)

            assert all(robot.disconnects == 1 for robot in first.values())
            assert all(robot.connected for robot in factory.robots.values())

    async def test_a_failed_load_leaves_the_session_empty(self) -> None:
        environment, factory = _teleop_env()
        factory.joints = {"follower": JOINTS, "leader": ("only_one",)}

        with pytest.raises(ValueError, match="same number of joints"):
            async with RuntimeSession(factory) as session:
                await session.load(environment)

        assert all(not robot.connected for robot in factory.robots.values())

    async def test_a_failed_load_does_not_keep_the_previous_environment(self) -> None:
        """The previous one is unloaded first, so a failure cannot silently
        leave the client driving the environment it asked to replace."""
        good, factory = _teleop_env()
        broken = _Environment(
            name="broken", robots=[_Configured(robot=_Row("follower"), tele_operator=_Teleoperator(robot=_Row("odd")))]
        )
        factory.roles = {"follower": "follower", "leader": "leader", "odd": "leader"}

        async with RuntimeSession(factory) as session:
            await session.load(good)
            factory.joints = {"odd": ("only_one",)}

            with pytest.raises(ValueError, match="same number of joints"):
                await session.load(broken)

            assert session.environment is None


class TestCommandsNeedAnEnvironment:
    async def test_teleoperating_without_a_loaded_environment_is_refused(self) -> None:
        async with RuntimeSession(_Factory()) as session:
            with pytest.raises(RuntimeError, match="No environment is loaded"):
                session.require_environment()

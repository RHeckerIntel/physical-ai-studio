"""An environment's shape comes through the driver layer, with nothing attached."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import pytest

from runtimev2.environment import describe_environment

JOINTS = ("shoulder_pan", "elbow_flex", "gripper")


@dataclass
class _Driver:
    joint_names: tuple[str, ...] = JOINTS
    connected: bool = False

    def connect(self) -> None:
        self.connected = True


@dataclass
class _Definition:
    role: str


@dataclass
class _Robot:
    name: str
    type: str = "SO101_Follower"
    id: Any = field(default_factory=uuid4)
    connection_string: str = "/dev/ttyACM0"


@dataclass
class _Camera:
    name: str
    width: int = 640
    height: int = 480
    id: Any = field(default_factory=uuid4)

    @property
    def payload(self) -> _Camera:
        return self


@dataclass
class _Teleoperator:
    robot: _Robot | None = None


@dataclass
class _Configured:
    robot: _Robot
    tele_operator: _Teleoperator = field(default_factory=_Teleoperator)


@dataclass
class _Environment:
    robots: list[_Configured] = field(default_factory=list)
    cameras: list[_Camera] = field(default_factory=list)


@dataclass
class _Factory:
    """Stands in for RobotClientFactory, recording what it was asked to build."""

    roles: dict[str, str] = field(default_factory=dict)
    joints: dict[str, tuple[str, ...]] = field(default_factory=dict)
    built: list[str] = field(default_factory=list)
    drivers: list[_Driver] = field(default_factory=list)

    async def build_robot_driver(self, robot: _Robot, port_finder: object) -> tuple[_Driver, _Definition]:
        self.built.append(robot.name)
        driver = _Driver(joint_names=self.joints.get(robot.name, JOINTS))
        self.drivers.append(driver)
        return driver, _Definition(role=self.roles.get(robot.name, "follower"))

    async def find_port(self, port_info: object) -> str | None:
        return None


async def test_a_robots_joints_come_from_its_driver() -> None:
    environment = _Environment(robots=[_Configured(robot=_Robot("Follower Arm"))])
    factory = _Factory(joints={"Follower Arm": ("a", "b")})

    shape = await describe_environment(environment, factory)

    assert len(shape.robots) == 1
    assert shape.robots[0].joint_names == ("a", "b")
    assert shape.robots[0].role == "follower"


async def test_nothing_is_connected() -> None:
    """The shape is needed while the user is still choosing, before anything is plugged in."""
    environment = _Environment(robots=[_Configured(robot=_Robot("follower"))])
    factory = _Factory()

    await describe_environment(environment, factory)

    assert factory.drivers
    assert not any(driver.connected for driver in factory.drivers)


async def test_a_teleoperator_is_described_as_a_leader() -> None:
    """A leader contributes observations, but is not one of the driven robots."""
    environment = _Environment(
        robots=[_Configured(robot=_Robot("follower"), tele_operator=_Teleoperator(robot=_Robot("leader")))]
    )
    factory = _Factory(roles={"follower": "follower", "leader": "leader"})

    shape = await describe_environment(environment, factory)

    assert [robot.key for robot in shape.robots] == ["follower"]
    assert [leader.key for leader in shape.leaders] == ["leader"]
    # No action slots for a leader: nothing commands one.
    spec = shape.feature_spec()
    assert "action.leader.shoulder_pan.pos" not in spec
    assert "observation.leader.shoulder_pan.pos" in spec


async def test_a_teleoperator_of_none_adds_nothing() -> None:
    environment = _Environment(robots=[_Configured(robot=_Robot("follower"))])

    shape = await describe_environment(environment, _Factory())

    assert len(shape.robots) == 1


async def test_display_names_are_sanitized_into_keys() -> None:
    environment = _Environment(
        robots=[_Configured(robot=_Robot("Follower Arm #2"))],
        cameras=[_Camera("Overhead Cam")],
    )

    shape = await describe_environment(environment, _Factory())

    assert shape.robots[0].key == "follower_arm_2"
    assert shape.cameras[0].key == "overhead_cam"


async def test_cameras_carry_their_declared_resolution() -> None:
    environment = _Environment(cameras=[_Camera("overhead", width=1280, height=720)])

    shape = await describe_environment(environment, _Factory())

    assert shape.cameras[0].shape == (720, 1280, 3)


async def test_names_that_collide_after_sanitizing_are_refused() -> None:
    """Merging them would hide two robots' joints under one key."""
    environment = _Environment(robots=[_Configured(robot=_Robot("Arm A")), _Configured(robot=_Robot("arm a"))])

    with pytest.raises(ValueError, match="collide after sanitizing"):
        await describe_environment(environment, _Factory())


async def test_the_shape_projects_onto_the_feature_spec() -> None:
    environment = _Environment(
        robots=[_Configured(robot=_Robot("follower"))],
        cameras=[_Camera("overhead")],
    )

    spec = (await describe_environment(environment, _Factory())).feature_spec()

    assert "observation.follower.gripper.pos" in spec
    assert "action.follower.gripper.pos" in spec
    assert spec["observation.images.overhead"].shape == (480, 640, 3)

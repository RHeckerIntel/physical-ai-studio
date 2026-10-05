"""A device set is what to open, whether or not an environment was saved."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

import pytest

from runtimev2.devices import DeviceSet, from_environment

FOLLOWER_ID = uuid4()
LEADER_ID = uuid4()
CAMERA_ID = uuid4()


@dataclass
class _Row:
    name: str
    id: object = field(default_factory=uuid4)


@dataclass
class _Teleoperator:
    robot: _Row


@dataclass
class _Configured:
    robot: _Row
    tele_operator: _Teleoperator | None = None


@dataclass
class _Environment:
    name: str = "test env"
    robots: list[_Configured] = field(default_factory=list)
    cameras: list[_Row] = field(default_factory=list)


class TestFromEnvironment:
    def test_a_teleoperator_becomes_a_leader(self) -> None:
        """Which is what the pairing in an environment means."""
        follower, leader = _Row("follower", FOLLOWER_ID), _Row("leader", LEADER_ID)
        environment = _Environment(robots=[_Configured(robot=follower, tele_operator=_Teleoperator(robot=leader))])

        devices = from_environment(environment)  # type: ignore[arg-type]

        assert devices.robots == (follower,)
        assert devices.leaders == (leader,)

    def test_a_robot_without_a_teleoperator_contributes_no_leader(self) -> None:
        environment = _Environment(robots=[_Configured(robot=_Row("follower"))])

        assert from_environment(environment).leaders == ()  # type: ignore[arg-type]

    def test_the_name_is_carried_for_logs_and_clients(self) -> None:
        environment = _Environment(name="assembly line")

        assert from_environment(environment).name == "assembly line"  # type: ignore[arg-type]

    def test_cameras_are_carried_as_rows(self) -> None:
        """Rows, because opening one needs its fingerprint and resolution."""
        camera = _Row("overhead", CAMERA_ID)
        environment = _Environment(cameras=[camera])

        assert from_environment(environment).cameras == (camera,)  # type: ignore[arg-type]


class TestLookups:
    def test_a_driven_robot_is_found_by_id(self) -> None:
        follower = _Row("follower", FOLLOWER_ID)
        devices = DeviceSet(name="rig", robots=(follower,))  # type: ignore[arg-type]

        assert devices.robot_row(str(FOLLOWER_ID)) is follower

    def test_a_leader_is_found_by_id_too(self) -> None:
        """A control needs the row to build the driver it reads."""
        leader = _Row("leader", LEADER_ID)
        devices = DeviceSet(name="rig", leaders=(leader,))  # type: ignore[arg-type]

        assert devices.robot_row(str(LEADER_ID)) is leader

    def test_an_unknown_robot_names_the_set(self) -> None:
        devices = DeviceSet(name="rig")

        with pytest.raises(RuntimeError, match="is not part of rig"):
            devices.robot_row(str(uuid4()))

    def test_a_camera_is_found_by_id(self) -> None:
        camera = _Row("overhead", CAMERA_ID)
        devices = DeviceSet(name="rig", cameras=(camera,))  # type: ignore[arg-type]

        assert devices.camera_row(str(CAMERA_ID)) is camera

    def test_an_unknown_camera_names_the_set(self) -> None:
        with pytest.raises(RuntimeError, match="is not part of rig"):
            DeviceSet(name="rig").camera_row(str(uuid4()))


def test_an_unsaved_set_needs_no_environment() -> None:
    """The setup wizard verifies a robot registered moments ago."""
    follower = _Row("fresh arm", FOLLOWER_ID)

    devices = DeviceSet(name=follower.name, robots=(follower,))  # type: ignore[arg-type]

    assert devices.name == "fresh arm"
    assert devices.leaders == ()
    assert devices.cameras == ()

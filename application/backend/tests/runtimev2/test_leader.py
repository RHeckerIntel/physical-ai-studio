"""A leader is a device we read, held outside any one control."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from runtimev2.environment import LeaderShape
from runtimev2.leader import LeaderDevice

JOINTS = ("shoulder_pan", "gripper")
SHAPE = LeaderShape(key="leader", robot_id="r0", joint_names=JOINTS)


@dataclass
class _Observation:
    joint_positions: np.ndarray
    timestamp: float


@dataclass
class _Driver:
    reported_joints: tuple[str, ...] = JOINTS
    connected: bool = False
    disconnects: int = 0
    reads: int = 0
    positions: np.ndarray = field(default_factory=lambda: np.array([1.0, 2.0], dtype=np.float32))

    @property
    def joint_names(self) -> list[str]:
        return list(self.reported_joints)

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False
        self.disconnects += 1

    def get_observation(self) -> _Observation:
        self.reads += 1
        return _Observation(joint_positions=self.positions, timestamp=7.5)


class TestLifecycle:
    async def test_entering_connects_and_leaving_releases(self) -> None:
        driver = _Driver()

        async with LeaderDevice(driver, SHAPE):
            assert driver.connected

        assert not driver.connected
        assert driver.disconnects == 1

    async def test_a_device_swapped_after_describing_is_refused(self) -> None:
        """Its positions would be mapped onto the wrong joints of the follower."""
        driver = _Driver(reported_joints=("gripper", "shoulder_pan"))

        with pytest.raises(RuntimeError, match="once connected"):
            async with LeaderDevice(driver, SHAPE):
                pass

    async def test_a_refused_check_still_releases_the_arm(self) -> None:
        driver = _Driver(reported_joints=("only_one",))

        with pytest.raises(RuntimeError):
            async with LeaderDevice(driver, SHAPE):
                pass

        assert driver.disconnects == 1


class TestReading:
    async def test_it_reports_positions_and_capture_time(self) -> None:
        device = LeaderDevice(_Driver(), SHAPE)

        async with device:
            values, timestamp = device.read()

        np.testing.assert_allclose(values, [1.0, 2.0])
        assert timestamp == 7.5

    def test_reading_before_connecting_is_refused(self) -> None:
        with pytest.raises(RuntimeError, match="not connected"):
            LeaderDevice(_Driver(), SHAPE).read()

    async def test_reading_after_release_is_refused(self) -> None:
        """A control outliving its leader must fail loudly, not read a closed port."""
        device = LeaderDevice(_Driver(), SHAPE)
        async with device:
            device.read()

        with pytest.raises(RuntimeError, match="not connected"):
            device.read()

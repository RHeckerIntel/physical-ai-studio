from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING

import numpy as np
import pytest

from runtime_session.controls import MoveControl, TeleopControl, build_control
from runtime_session.store import ACTION, STATE, FeatureSpec, Store
from schemas.runtime_session import MoveControlConfig, TeleopControlConfig
from tests.workers.fake_robot import FakeRobot
from workers.base import WorkerStartupError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

JOINTS = ("shoulder_pan.pos", "gripper.pos")


@pytest.fixture
def store() -> Iterator[Store]:
    store = Store.create(FeatureSpec(JOINTS), uuid.uuid4().hex)
    store.write(STATE, [0.0, 10.0], timestamp=time.monotonic())
    yield store
    store.unlink()


def _leader(tmp_path: Path, joints: list[str] | None = None) -> FakeRobot:
    leader = FakeRobot(str(tmp_path / "leader"), joints or ["shoulder_pan", "gripper"], [5.0, 6.0])
    leader.connect()
    return leader


def test_teleop_commands_what_the_leader_measures(store: Store, tmp_path: Path) -> None:
    control = TeleopControl(store, _leader(tmp_path), TeleopControlConfig(grace_s=0))
    control.on_start()
    control.on_tick()
    action = store.read(ACTION)
    assert action.value.tolist() == [5.0, 6.0]
    assert action.timestamp <= time.monotonic()


def test_teleop_closes_the_gap_to_the_leader_over_the_grace_period(store: Store, tmp_path: Path) -> None:
    control = TeleopControl(store, _leader(tmp_path), TeleopControlConfig(grace_s=1.0))
    control.on_start()
    control.on_tick()
    # Starts where the follower is, not where the leader is.
    np.testing.assert_allclose(store.read(ACTION).value, [0.0, 10.0], atol=0.1)

    control.started_at = time.monotonic() - 0.5  # halfway through
    control.on_tick()
    np.testing.assert_allclose(store.read(ACTION).value, [2.5, 8.0], atol=0.1)

    control.started_at = time.monotonic() - 2.0  # past it: on the leader
    control.on_tick()
    assert store.read(ACTION).value.tolist() == [5.0, 6.0]


async def test_every_start_of_teleop_has_its_own_grace_period(store: Store, tmp_path: Path) -> None:
    leader = _leader(tmp_path)
    for _ in range(2):
        store.write(STATE, [0.0, 10.0], timestamp=time.monotonic())
        async with TeleopControl(store, leader, TeleopControlConfig(grace_s=5.0)):
            await asyncio.sleep(0.05)
        # A fresh control blends from the follower again rather than jumping to the leader.
        assert store.read(ACTION).value[0] < 1.0


def test_teleop_refuses_a_leader_with_other_joints(store: Store, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="do not match"):
        TeleopControl(store, _leader(tmp_path, ["elbow", "gripper"]), TeleopControlConfig())


def test_teleop_needs_a_leader(store: Store) -> None:
    with pytest.raises(ValueError, match="leader arm"):
        build_control(TeleopControlConfig(), store, None)


def test_move_interpolates_then_holds_the_target(store: Store) -> None:
    control = MoveControl(store, MoveControlConfig(position={"shoulder_pan.pos": 4.0}, duration_s=0.2))
    control.on_start()
    control.started_at = time.monotonic() - 0.1  # halfway
    control.on_tick()
    np.testing.assert_allclose(store.read(ACTION).value, [2.0, 10.0], atol=0.2)
    control.started_at = time.monotonic() - 1.0  # long done
    control.on_tick()
    # The joint left out stays where it was.
    assert store.read(ACTION).value.tolist() == [4.0, 10.0]


async def test_move_fails_to_start_before_the_follower_is_measured() -> None:
    unmeasured = Store.create(FeatureSpec(JOINTS), uuid.uuid4().hex)
    try:
        with pytest.raises(WorkerStartupError, match="not been measured"):
            async with build_control(MoveControlConfig(position={}, duration_s=1), unmeasured, None):
                pass
    finally:
        unmeasured.unlink()


def test_move_refuses_unknown_joints(store: Store) -> None:
    with pytest.raises(ValueError, match="Unknown joints"):
        build_control(MoveControlConfig(position={"elbow.pos": 1.0}, duration_s=1), store, None)


async def test_a_control_runs_as_a_thread_until_stopped(store: Store, tmp_path: Path) -> None:
    async with TeleopControl(store, _leader(tmp_path), TeleopControlConfig()):
        await asyncio.sleep(0.1)
    first = store.read(ACTION).timestamp
    await asyncio.sleep(0.05)
    assert store.read(ACTION).timestamp == first  # stopped writing

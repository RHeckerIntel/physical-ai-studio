from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

from runtime_session.bus import BusRequestError, command_key
from runtime_session.session import RuntimeSession
from runtime_session.store import ACTION, STATE
from schemas.runtime_session import (
    CommandDone,
    HilControlConfig,
    InterruptCommand,
    ModelControlConfig,
    ResumeCommand,
    TakeOverCommand,
)
from tests.workers.fake_dataset import FakeDatasetWorker
from tests.workers.fake_model import FakeModelWorker
from tests.workers.fake_robot import FakeRobot, FakeTorqueRobot, sent_actions

from .fixtures import JOINTS, LEADER_ID, make_environment
from .test_model import _model, _until
from .test_recording import _dataset

if TYPE_CHECKING:
    from pydantic import BaseModel


@pytest.fixture
def fake_model_worker():
    with patch("runtime_session.model.ModelWorker", FakeModelWorker):
        yield


@pytest.fixture
def fake_dataset_worker():
    with patch("runtime_session.dataset.DatasetWorker", FakeDatasetWorker):
        yield


async def _send(session: RuntimeSession, command: BaseModel) -> None:
    await session.bus.request(command_key(command.command), command, CommandDone, timeout=30)  # type: ignore[attr-defined]


def _phase(session: RuntimeSession) -> str:
    return session.state().control.phase  # type: ignore[union-attr]


async def test_interrupt_take_over_and_resume(
    new_session, fake_cameras, fake_model_worker, fake_dataset_worker, tmp_path
) -> None:
    session: RuntimeSession = await new_session()
    await session.load_environment(make_environment())
    await session.load_model(_model(tmp_path), "torch", "cpu")
    await session.load_dataset(_dataset())
    store = session.store
    try:
        await session.set_control(HilControlConfig(model=ModelControlConfig(task="pick", hz=50, grace_s=0)))
        assert _phase(session) == "inference"
        start = store.read(STATE).value.copy()
        await _until(lambda: store.read(STATE).value[0] > start[0] + 3)  # the model drives

        with pytest.raises(BusRequestError, match="inference"):
            await _send(session, TakeOverCommand(request_id=""))

        await _send(session, InterruptCommand(request_id=""))
        await _until(lambda: _phase(session) == "interrupted")
        held = store.read(ACTION).value.copy()
        await _until(lambda: (store.read(STATE).value == held).all())  # the model no longer drives

        await _send(session, TakeOverCommand(request_id=""))
        await _until(lambda: _phase(session) == "teleop" and session.state().dataset.is_recording)

        await asyncio.sleep(0.3)  # record a few frames
        await _send(session, ResumeCommand(request_id=""))
        await _until(lambda: _phase(session) == "inference" and session.state().dataset.episodes_recorded == 1)
        assert not session.state().dataset.is_recording
    finally:
        await session.stop()


async def test_take_over_needs_a_dataset(new_session, fake_cameras, fake_model_worker, tmp_path) -> None:
    session: RuntimeSession = await new_session()
    await session.load_environment(make_environment())
    await session.load_model(_model(tmp_path), "torch", "cpu")
    try:
        await session.set_control(HilControlConfig(model=ModelControlConfig(task="pick", hz=50)))
        await _send(session, InterruptCommand(request_id=""))
        with pytest.raises(BusRequestError, match="Load a dataset"):
            await _send(session, TakeOverCommand(request_id=""))
        await _send(session, ResumeCommand(request_id=""))  # back to the model, nothing recorded
        await _until(lambda: _phase(session) == "inference")
    finally:
        await session.stop()


async def test_hil_needs_a_leader(new_session, fake_cameras, fake_model_worker, tmp_path) -> None:
    session: RuntimeSession = await new_session()
    await session.load_environment(make_environment(leader=False))
    await session.load_model(_model(tmp_path), "torch", "cpu")
    try:
        with pytest.raises(ValueError, match="leader arm"):
            await session.set_control(HilControlConfig(model=ModelControlConfig()))
    finally:
        await session.stop()


async def test_interrupt_moves_a_torque_leader_to_the_follower(
    new_session, fake_cameras, fake_model_worker, robot_factory, tmp_path
) -> None:
    async def build(robot, _port_finder):
        if robot.id == LEADER_ID:
            return FakeTorqueRobot(str(tmp_path / "leader.jsonl"), JOINTS, [30.0, 40.0]), object()
        return FakeRobot(str(tmp_path / "follower.jsonl"), JOINTS, [1.0, 2.0]), object()

    robot_factory.build_robot_driver.side_effect = build
    session: RuntimeSession = await new_session()
    await session.load_environment(make_environment())
    await session.load_model(_model(tmp_path), "torch", "cpu")
    leader = session.leader
    assert isinstance(leader, FakeTorqueRobot)
    try:
        config = HilControlConfig(model=ModelControlConfig(task="pick", hz=50, grace_s=0), leader_align_s=0.3)
        await session.set_control(config)
        await _send(session, InterruptCommand(request_id=""))
        held = session.store.read(ACTION).value.copy()
        assert leader.torque
        # Stepped there over leader_align_s, not sent in one jump.
        await _until(lambda: np.allclose(leader.get_observation().joint_positions, held))
        assert len(sent_actions(tmp_path / "leader.jsonl")) > 3

        await _send(session, ResumeCommand(request_id=""))
        assert not leader.torque  # free again for the human
    finally:
        await session.stop()

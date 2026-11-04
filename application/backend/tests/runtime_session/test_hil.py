from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

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

from .fixtures import make_environment
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


class _TrossenSdk:
    """Records what the leader's SDK is told, posing as the Trossen one."""

    __module__ = "trossen_arm.trossen_arm"

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def set_all_modes(self, mode) -> None:
        self.calls.append(("mode", mode))

    def set_all_positions(self, positions, goal_time, blocking) -> None:
        self.calls.append(("positions", list(positions), goal_time, blocking))

    def set_all_external_efforts(self, efforts, goal_time, blocking) -> None:
        self.calls.append(("efforts", list(efforts), goal_time, blocking))


def test_a_trossen_leader_moves_in_radians_and_is_released() -> None:
    import numpy as np
    import trossen_arm

    from runtime_session.controls.leader import move_leader, release_leader

    leader = SimpleNamespace(joint_names=["joint_0", "gripper"], _driver=_TrossenSdk())
    assert move_leader(leader, np.array([180.0, 0.04]), duration=2.0)  # type: ignore[arg-type]
    release_leader(leader)  # type: ignore[arg-type]
    assert leader._driver.calls == [
        ("mode", trossen_arm.Mode.position),
        ("positions", [pytest.approx(np.pi), pytest.approx(0.04)], 2.0, False),
        ("mode", trossen_arm.Mode.external_effort),
        ("efforts", [0.0, 0.0], 0.0, False),
    ]
    # Any other leader is left for the human to bring over.
    assert not move_leader(SimpleNamespace(joint_names=["joint_0"]), np.array([1.0]), duration=2.0)  # type: ignore[arg-type]

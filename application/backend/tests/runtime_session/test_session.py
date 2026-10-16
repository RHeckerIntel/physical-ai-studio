from __future__ import annotations

import asyncio
import time

import pytest

from exceptions import DeviceInUseError, RuntimeSessionCommandError
from runtime_session.store import ACTION, STATE
from workers.base import WorkerStartupError

from .fixtures import FOLLOWER_ID, LEADER_ID, make_environment


async def _until(condition, within: float = 5.0) -> None:
    deadline = time.monotonic() + within
    while not condition():
        assert time.monotonic() < deadline, "condition never held"
        await asyncio.sleep(0.02)


async def test_loading_connects_everything_and_publishes_state(new_session, registry, fake_cameras) -> None:
    session = new_session()
    try:
        await session.load_environment(make_environment())

        environment = session.environment
        assert environment is not None
        assert environment.joint_names == ["shoulder_pan.pos", "gripper.pos"]
        assert [camera.key for camera in environment.cameras] == ["cam 0"]
        assert session.leader is not None and session.leader.is_connected()
        assert registry.holder_of_robot(FOLLOWER_ID) is session
        assert registry.holder_of_robot(LEADER_ID) is session
        assert session.events.get_nowait().data.environment == environment

        store = session.store
        await _until(lambda: store.read(STATE) is not None and store.read_observation("cam 0") is not None)
        assert store.read(ACTION).value.tolist() == [1.0, 2.0]  # seeded: the arm holds still
    finally:
        await session.close("socket_closed")


async def test_unloading_releases_everything(new_session, registry, camera_claims, fake_cameras) -> None:
    session = new_session()
    environment = make_environment()
    await session.load_environment(environment)
    leader = session.leader
    await session.unload_environment()

    assert session.environment is None and session.store is None
    assert leader is not None and not leader.is_connected()
    assert registry.holder_of_robot(FOLLOWER_ID) is None
    assert camera_claims.holder_of(environment.cameras[0].fingerprint) is None
    # And it can load again.
    await session.load_environment(environment)
    await session.close("socket_closed")
    assert registry.get(session.id) is None


async def test_one_environment_at_a_time(new_session, fake_cameras) -> None:
    session = new_session()
    await session.load_environment(make_environment())
    try:
        with pytest.raises(RuntimeSessionCommandError):
            await session.load_environment(make_environment())
    finally:
        await session.close("socket_closed")


async def test_another_session_cannot_take_the_robot(new_session, fake_cameras) -> None:
    first, second = new_session(), new_session()
    await first.load_environment(make_environment())
    try:
        with pytest.raises(DeviceInUseError, match="Bench"):
            await second.load_environment(make_environment(cameras=0))
        assert second.environment is None
    finally:
        await first.close("socket_closed")
    await second.load_environment(make_environment(cameras=0))
    await second.close("socket_closed")


async def test_a_failed_connect_leaves_nothing_held(new_session, registry, robot_factory, tmp_path) -> None:
    from tests.workers.fake_robot import FakeRobot

    async def build(robot, _port_finder):
        failing = robot.id == FOLLOWER_ID
        return FakeRobot(str(tmp_path / f"{robot.id}"), ["shoulder_pan"], [0.0], fail_connect=failing), object()

    robot_factory.build_robot_driver.side_effect = build
    session = new_session()
    with pytest.raises(WorkerStartupError, match="Failed to open serial port"):
        await session.load_environment(make_environment(cameras=0))
    assert session.environment is None
    assert registry.holder_of_robot(LEADER_ID) is None
    await session.close("socket_closed")


async def test_a_dead_worker_unloads_the_environment_but_not_the_session(new_session, fake_cameras) -> None:
    session = new_session()
    await session.load_environment(make_environment())
    session.events.get_nowait()
    fake_cameras.return_value.read_latest.side_effect = RuntimeError("camera unplugged")

    error = await asyncio.wait_for(session.events.get(), 5)
    assert error.type == "error"
    assert error.error_code == "worker_failed"
    assert error.session_alive
    assert "camera unplugged" in error.message
    state = await asyncio.wait_for(session.events.get(), 15)
    assert state.type == "state" and state.data.environment is None
    assert not session.closed.is_set()
    await session.close("socket_closed")


async def test_stopping_tells_the_socket(new_session) -> None:
    session = new_session()
    await session.close("stopped")
    event = session.events.get_nowait()
    assert event.error_code == "session_closed"
    assert not event.session_alive

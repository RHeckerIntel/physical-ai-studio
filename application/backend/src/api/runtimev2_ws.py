"""Websocket for the new runtime: load an environment, watch it, drive it.

Deliberately separate from ``runtime_ws``. That endpoint is what the UI uses
today and it works; this one exists to find out whether the store-and-workers
pattern holds, which means being free to change shape without breaking
anything people depend on.

Thin on purpose. The session owns the environment, the store and the workers,
so this only translates messages into method calls and reports what the store
already holds.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any
from uuid import UUID  # noqa: TC003  # FastAPI evaluates websocket annotations at runtime

from fastapi import APIRouter, Depends, WebSocket, status
from fastapi.responses import Response
from fastapi.websockets import WebSocketDisconnect
from loguru import logger

from api.dependencies import (
    CameraClaimRegistryDep,
    DatasetServiceDep,
    EnvironmentServiceDep,
    ProjectCameraServiceDep,
    ProjectServiceDep,
    RobotClientFactoryDep,
    RobotServiceDep,
    get_camera_id,
    get_dataset_id,
    get_environment_id,
    get_model_id,
    get_project_id,
    get_robot_id,
)
from exceptions import BaseException as AppBaseException
from runtimev2.control.config import describe as describe_control
from runtimev2.control.config import parse as parse_control
from runtimev2.devices import DeviceSet, from_environment
from runtimev2.inference import export_dir
from runtimev2.pinning import CameraPinning
from runtimev2.session import DEFAULT_DATASET_HZ, RuntimeSession
from schemas.hardware import InferenceDevice
from settings import get_settings

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from runtimev2.session_store import SessionStore
    from services import ProjectCameraService, RobotService
    from services.dataset_service import DatasetService
    from services.environment_service import EnvironmentService

router = APIRouter(prefix="/api/projects/{project_id}/runtimev2", tags=["Runtime v2"])

# Observations are a view, not a control path: fast enough to look live, slow
# enough that a 100Hz robot does not flood a browser with 100 messages a second.
_STREAM_HZ = 30.0


def _state_message(session: RuntimeSession) -> dict[str, Any]:
    """Describe the session, loaded or not.

    ``loaded: false`` is a normal state, not an error: a session with nothing
    loaded is what a client sees before its first load and after an unload.
    """
    state = session.state()
    loaded = state.environment
    return {
        "event": "state",
        "data": {
            "loaded": state.loaded,
            "environment": loaded.environment if loaded else None,
            "robots": loaded.robots if loaded else {},
            "cameras": list(loaded.cameras) if loaded else [],
            "leaders": list(loaded.leaders) if loaded else [],
            "control": describe_control(state.control) if state.control else None,
            "features": loaded.features if loaded else 0,
            "dataset_loaded": state.dataset_loaded,
            "dataset_id": state.dataset_id,
            "dataset_hz": state.dataset_hz,
            "is_recording": state.is_recording,
            "episodes_recorded": state.episodes_recorded,
            "task": state.task,
            "model_loaded": state.model_loaded,
            "model_id": state.model_id,
            "model_chunk_size": state.model_chunk_size,
        },
    }


def _ack_message(request_id: str, *, exc: Exception | None = None) -> dict[str, Any]:
    """Answer one identified command, after its effect is already reported.

    Sent after the ``state`` the command produced, so a client awaiting it has
    the resulting state in hand by the time it resolves.
    """
    data: dict[str, Any] = {"request_id": request_id, "ok": exc is None}
    if exc is not None:
        data["error"] = exc.message if isinstance(exc, AppBaseException) else str(exc)
        data["error_code"] = exc.error_code if isinstance(exc, AppBaseException) else "runtime_command_failed"
    return {"event": "ack", "data": data}


def _error_message(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, AppBaseException):
        return {"event": "error", "message": exc.message, "error_code": exc.error_code}
    return {"event": "error", "message": str(exc), "error_code": "runtime_session_failed"}


def _observation_message(store: SessionStore) -> dict[str, Any]:
    """Send every scalar feature the store currently holds.

    Images are left out: they are not in the store yet, and when they are they
    will not travel as JSON.
    """
    snapshot = store.snapshot(store.spec.keys())
    return {
        "event": "observation",
        "data": {
            key: {"value": sample.value, "timestamp": sample.timestamp}
            for key, sample in snapshot.items()
            if not store.spec[key].is_image
        },
    }


async def _handle_incoming(  # the services a command may need
    websocket: WebSocket,
    session: RuntimeSession,
    environment_service: EnvironmentService,
    dataset_service: DatasetService,
    robot_service: RobotService,
    camera_service: ProjectCameraService,
    project_id: UUID,
) -> None:
    """Translate client messages into calls on the session.

    Returns on an explicit ``disconnect``, or when the socket closes. A command
    that fails is reported and the session carries on -- asking to load an
    environment whose robot is unplugged should not drop the connection.

    A command carrying ``request_id`` is answered with an ``ack`` bearing the
    same id. Nothing else correlates a reply with the command that caused it:
    the stream is one-way, so without an id a client watching ``state`` cannot
    tell its own save from someone else's, or a slow one from a failed one.
    An unknown command is ignored when nothing is waiting on it and refused by
    ack when something is, so an awaited request always gets an answer.
    """
    try:
        while True:
            message = await websocket.receive_json("text")
            event = message.get("event")
            if event == "disconnect":
                return
            request_id = message.get("request_id")
            request_id = str(request_id) if request_id is not None else None
            try:
                handled = await _apply(
                    websocket,
                    session,
                    environment_service,
                    dataset_service,
                    robot_service,
                    camera_service,
                    project_id,
                    message,
                )
            except Exception as exc:
                logger.warning("runtimev2 command {} failed: {}", event, exc)
                # Reported once, to whoever asked: an identified request gets
                # an identified answer rather than that plus a loose error.
                if request_id is None:
                    await websocket.send_json(_error_message(exc))
                else:
                    await websocket.send_json(_ack_message(request_id, exc=exc))
            else:
                if request_id is not None:
                    await websocket.send_json(
                        _ack_message(request_id)
                        if handled
                        else _ack_message(request_id, exc=ValueError(f"Unknown command {event!r}"))
                    )
    except WebSocketDisconnect:
        logger.debug("runtimev2 websocket closed; ending the session")


@dataclass(frozen=True, slots=True)
class _Command:
    """One client message, with everything a handler might need to serve it."""

    session: RuntimeSession
    environment_service: EnvironmentService
    dataset_service: DatasetService
    robot_service: RobotService
    camera_service: ProjectCameraService
    project_id: UUID
    message: dict[str, Any]

    def arg(self, name: str) -> Any:
        return self.message.get(name)


async def _load_environment(command: _Command) -> None:
    environment = await command.environment_service.get_environment_by_id(
        command.project_id, get_environment_id(str(command.arg("environment_id")))
    )
    await command.session.load(from_environment(environment))


async def _load_devices(command: _Command) -> None:
    """Open an explicit set of devices, with no environment behind it.

    The setup wizard verifies a robot registered moments ago, so there is no
    environment to name and there cannot be one.
    """
    follower = await command.robot_service.get_robot_by_id(
        command.project_id, get_robot_id(str(command.arg("follower_id")))
    )
    leaders = []
    if command.arg("leader_id") is not None:
        leaders.append(
            await command.robot_service.get_robot_by_id(command.project_id, get_robot_id(str(command.arg("leader_id"))))
        )
    cameras = []
    for raw in command.arg("camera_ids") or []:
        cameras.append(await command.camera_service.get_camera_by_id(command.project_id, get_camera_id(str(raw))))
    await command.session.load(
        DeviceSet(
            # Named for the robot being driven: there is no environment name to
            # borrow, and a client showing this wants to know which arm it is.
            name=follower.name,
            robots=(follower,),
            leaders=tuple(leaders),
            cameras=tuple(cameras),
        )
    )


async def _unload_environment(command: _Command) -> None:
    await command.session.unload()


async def _set_control(command: _Command) -> None:
    """Choose what drives the followers.

    Takes a config -- ``{"kind": "teleop", "hz": 100}`` -- or a bare kind when
    the default rate will do, or null to drive with nothing.
    """
    await command.session.select_control(parse_control(command.arg("control")))


async def _load_dataset(command: _Command) -> None:
    dataset = await command.dataset_service.get_dataset_by_id(get_dataset_id(str(command.arg("dataset_id"))))
    await command.session.load_dataset(
        dataset.id,
        get_settings().datasets_dir / str(dataset.id),
        hz=int(command.arg("hz") or DEFAULT_DATASET_HZ),
    )


async def _unload_dataset(command: _Command) -> None:
    await command.session.unload_dataset()


async def _start_recording(command: _Command) -> None:
    command.session.start_recording(str(command.arg("task")))


async def _save_episode(command: _Command) -> None:
    await command.session.save_episode()


async def _discard_episode(command: _Command) -> None:
    await command.session.discard_episode()


async def _set_task(command: _Command) -> None:
    task = command.arg("task")
    command.session.set_task(str(task) if task else None)


async def _load_model(command: _Command) -> None:
    # ``{"backend": "...", "device": "..."}``, as the hardware endpoints report it.
    selected = InferenceDevice.model_validate(command.arg("inference_device"))
    model_id = get_model_id(str(command.arg("model_id")))
    await command.session.load_policy(
        model_id,
        export_dir(get_settings().models_dir, model_id, selected.backend.value),
        device=selected.device,
    )


async def _unload_model(command: _Command) -> None:
    await command.session.unload_policy()


_HANDLERS: dict[str, Callable[[_Command], Awaitable[None]]] = {
    "load_environment": _load_environment,
    "load_devices": _load_devices,
    "unload_environment": _unload_environment,
    "set_control": _set_control,
    "load_dataset": _load_dataset,
    "unload_dataset": _unload_dataset,
    "start_recording": _start_recording,
    "save_episode": _save_episode,
    "discard_episode": _discard_episode,
    "set_task": _set_task,
    "load_model": _load_model,
    "unload_model": _unload_model,
}


async def _apply(  # noqa: PLR0913, PLR0917  # the services a command may need
    websocket: WebSocket,
    session: RuntimeSession,
    environment_service: EnvironmentService,
    dataset_service: DatasetService,
    robot_service: RobotService,
    camera_service: ProjectCameraService,
    project_id: UUID,
    message: dict[str, Any],
) -> bool:
    """Run one client command and report the session's new state.

    Returns:
        Whether the command was one this build knows. An unknown one is
        ignored rather than closing the socket: this endpoint exists to change
        shape, so a client ahead of the server should not lose its connection
        for asking.
    """
    event = str(message.get("event"))
    handler = _HANDLERS.get(event)
    if handler is None:
        logger.debug("Ignoring unknown runtimev2 event {}", event)
        return False
    await handler(
        _Command(session, environment_service, dataset_service, robot_service, camera_service, project_id, message)
    )
    await websocket.send_json(_state_message(session))
    return True


async def _handle_outgoing(websocket: WebSocket, session: RuntimeSession) -> None:
    """Stream what the loaded environment holds, until the socket closes.

    Keeps running across loads and unloads: there is simply nothing to send
    while no environment is loaded.
    """
    period = 1.0 / _STREAM_HZ
    try:
        while True:
            loaded = session.environment
            if loaded is not None:
                await websocket.send_json(_observation_message(loaded.store))
            await asyncio.sleep(period)
    except WebSocketDisconnect:
        pass


@router.get("/ws", tags=["WebSocket"], summary="Runtime v2 session (WebSocket)", status_code=426)
async def runtimev2_websocket_openapi(project_id: UUID) -> Response:  # noqa: ARG001
    """This endpoint requires a WebSocket connection. Use `wss://` to connect."""
    return Response(status_code=426)


@router.websocket("/ws")
async def runtimev2_websocket(  # noqa: PLR0913, PLR0917  # one dependency per service a command needs
    project_id: Annotated[UUID, Depends(get_project_id)],
    environment_service: EnvironmentServiceDep,
    dataset_service: DatasetServiceDep,
    robot_client_factory: RobotClientFactoryDep,
    robot_service: RobotServiceDep,
    camera_service: ProjectCameraServiceDep,
    project_service: ProjectServiceDep,
    claims: CameraClaimRegistryDep,
    websocket: WebSocket,
) -> None:
    """Run one environment for as long as this websocket is open.

    Accepts ``{"event": "load_environment", "environment_id": "..."}``,
    ``{"event": "unload_environment"}``,
    ``{"event": "set_control", "control": {"kind": "teleop", "hz": 100}}`` and
    ``{"event": "disconnect"}``.

    Any command may carry a ``request_id``; it is answered with an ``ack``
    carrying the same id once the command has run.

    The session opens empty. Nothing is connected until an environment is
    loaded, and one can be swapped for another without reconnecting.
    """
    await websocket.accept()
    try:
        project = await project_service.get_project_by_id(project_id)
        pinning = CameraPinning(registry=claims, project_id=project_id, project_name=project.name)
        async with RuntimeSession(robot_client_factory, pinning) as session:
            await websocket.send_json(_state_message(session))
            incoming = asyncio.create_task(
                _handle_incoming(
                    websocket,
                    session,
                    environment_service,
                    dataset_service,
                    robot_service,
                    camera_service,
                    project_id,
                )
            )
            outgoing = asyncio.create_task(_handle_outgoing(websocket, session))
            try:
                done, pending = await asyncio.wait({incoming, outgoing}, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for task in done:
                    task.result()
            finally:
                for task in (incoming, outgoing):
                    task.cancel()
                await asyncio.gather(incoming, outgoing, return_exceptions=True)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        message = exc.message if isinstance(exc, AppBaseException) else str(exc)
        code = exc.error_code if isinstance(exc, AppBaseException) else "runtime_session_failed"
        logger.exception("runtimev2 websocket failed: {}", message)
        try:
            await websocket.send_json({"event": "error", "message": message, "error_code": code})
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception as close_exc:
            logger.error("Could not close runtimev2 websocket: {}", close_exc)

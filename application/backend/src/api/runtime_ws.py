"""The runtime session websocket: one session for the lifetime of the socket.

The client loads things into the session with commands, each acked. The server
sends the session's state on change, errors, observations at ``hz`` and every
new camera frame as a binary message. See ``schemas/runtime_session.py``.
"""

import asyncio
import json
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, WebSocket
from fastapi.responses import Response
from fastapi.websockets import WebSocketDisconnect
from loguru import logger
from pydantic import TypeAdapter, ValidationError

from api.dependencies import (
    CameraClaimRegistryDep,
    ProjectCameraServiceDep,
    ProjectServiceDep,
    RobotClientFactoryDep,
    RobotServiceDep,
    RuntimeSessionRegistryDep,
)
from exceptions import BaseException as AppBaseException
from runtime_session.session import RuntimeSession
from runtime_session.store import ACTION, STATE, Sample
from schemas.environment import (
    EnvironmentWithRelations,
    RobotWithTeleoperator,
    TeleoperatorNoneWithRobot,
    TeleoperatorRobotWithRobot,
)
from schemas.runtime_session import (
    AckMessage,
    EnvironmentDraft,
    LoadEnvironmentCommand,
    ObservationFrame,
    ObservationMessage,
    RuntimeSessionCommand,
    RuntimeSessionServerMessage,
    SessionError,
    StateMessage,
    UnloadEnvironmentCommand,
)
from services import RobotService
from services.project_camera_service import ProjectCameraService
from workers.base import WorkerStartupError
from workers.rate import Rate

router = APIRouter(prefix="/api/projects/{project_id}/runtime", tags=["Runtime Sessions"])

_COMMANDS: TypeAdapter[RuntimeSessionCommand] = TypeAdapter(RuntimeSessionCommand)


class _SessionEnded(Exception):
    pass


@router.get(
    "/ws",
    tags=["WebSocket"],
    summary="Runtime session (WebSocket)",
    status_code=426,
    responses={
        200: {"model": RuntimeSessionServerMessage, "description": "JSON messages the server sends"},
        299: {"model": RuntimeSessionCommand, "description": "JSON commands the client sends"},
    },
)
async def runtime_session_websocket_openapi(
    project_id: UUID,  # noqa: ARG001
    hz: Annotated[float, Query(gt=0, le=60, description="Observation and frame rate")] = 20,  # noqa: ARG001
) -> Response:
    """This endpoint requires a WebSocket connection. Use `wss://` to connect."""
    return Response(status_code=426)


@router.websocket("/ws")
async def runtime_session_websocket(  # noqa: PLR0913, D103 - FastAPI dependencies; documented on the GET stub
    websocket: WebSocket,
    project_id: UUID,
    project_service: ProjectServiceDep,
    robot_service: RobotServiceDep,
    camera_service: ProjectCameraServiceDep,
    robot_client_factory: RobotClientFactoryDep,
    registry: RuntimeSessionRegistryDep,
    camera_claims: CameraClaimRegistryDep,
    hz: Annotated[float, Query(gt=0, le=60)] = 20,
) -> None:
    await websocket.accept()
    project = await project_service.get_project_by_id(project_id)
    session = RuntimeSession(
        project_id=project_id,
        project_name=project.name,
        registry=registry,
        camera_claims=camera_claims,
        robot_factory=robot_client_factory,
    )
    registry.add(session)

    async def resolve(draft: EnvironmentDraft) -> EnvironmentWithRelations:
        return await _resolve_environment(draft, project_id, robot_service, camera_service)

    try:
        await websocket.send_json(StateMessage(data=session.state()).model_dump(mode="json"))
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(_receive_commands(websocket, session, resolve))
            tasks.create_task(_send_events(websocket, session))
            tasks.create_task(_sample_store(websocket, session, hz))
    except* (WebSocketDisconnect, _SessionEnded):
        pass
    except* Exception as errors:
        logger.opt(exception=errors).error("Runtime session {} websocket failed", session.id)
    finally:
        await session.close("socket_closed")


async def _resolve_environment(
    draft: EnvironmentDraft, project_id: UUID, robots: RobotService, cameras: ProjectCameraService
) -> EnvironmentWithRelations:
    """Look up the rows a draft refers to, in this project."""
    configured = []
    for robot in draft.robots:
        teleoperator: TeleoperatorRobotWithRobot | TeleoperatorNoneWithRobot = TeleoperatorNoneWithRobot()
        if robot.tele_operator.type == "robot":
            leader = await robots.get_robot_by_id(project_id, robot.tele_operator.robot_id)
            teleoperator = TeleoperatorRobotWithRobot(robot_id=leader.id, robot=leader)
        follower = await robots.get_robot_by_id(project_id, robot.robot_id)
        configured.append(RobotWithTeleoperator(robot=follower, tele_operator=teleoperator))
    return EnvironmentWithRelations(
        id=UUID(int=0),  # unused: a session does not care whether the environment is saved
        name=draft.name,
        robots=configured,
        cameras=[await cameras.get_camera_by_id(project_id, camera.camera_id) for camera in draft.cameras],
    )


async def _receive_commands(websocket: WebSocket, session: RuntimeSession, resolve) -> None:  # noqa: ANN001
    """Run commands one at a time, in order, and ack each."""
    while True:
        raw = await websocket.receive_text()
        try:
            command = _COMMANDS.validate_json(raw)
        except ValidationError as error:
            await websocket.send_json(
                AckMessage(
                    request_id=_request_id(raw),
                    ok=False,
                    error=SessionError(message=str(error), error_code="invalid_command"),
                ).model_dump(mode="json")
            )
            continue
        ack = await _run(command, session, resolve)
        await websocket.send_json(ack.model_dump(mode="json"))


def _request_id(raw: str) -> str:
    try:
        return str(json.loads(raw).get("request_id", ""))
    except (ValueError, AttributeError):
        return ""


async def _run(command: RuntimeSessionCommand, session: RuntimeSession, resolve) -> AckMessage:  # noqa: ANN001
    try:
        match command:
            case LoadEnvironmentCommand(environment=draft):
                await session.load_environment(await resolve(draft))
            case UnloadEnvironmentCommand():
                await session.unload_environment()
    except AppBaseException as error:
        return _failed(command, error.message, error.error_code)
    except WorkerStartupError as error:
        return _failed(command, f"{error.worker_name}: {error.reason}", "device_connection_failed")
    except ValueError as error:
        return _failed(command, str(error), "invalid_session_command")
    except Exception as error:
        logger.exception("Runtime session {} command {} failed", session.id, command.command)
        return _failed(command, str(error) or type(error).__name__, "internal_error")
    return AckMessage(request_id=command.request_id, ok=True)


def _failed(command: RuntimeSessionCommand, message: str, error_code: str) -> AckMessage:
    return AckMessage(
        request_id=command.request_id, ok=False, error=SessionError(message=message, error_code=error_code)
    )


async def _send_events(websocket: WebSocket, session: RuntimeSession) -> None:
    while True:
        event = await session.events.get()
        await websocket.send_json(event.model_dump(mode="json"))
        if event.type == "error" and not event.session_alive:
            await websocket.close()
            raise _SessionEnded


def _named(sample: Sample | None) -> dict[str, float] | None:
    return None if sample is None else dict(zip(sample.names, map(float, sample.value), strict=True))


async def _sample_store(websocket: WebSocket, session: RuntimeSession, hz: float) -> None:
    """Send the store at ``hz``: one observation per tick with the joints and every new camera frame."""
    from utils.jpeg import encode_jpeg_rgb

    rate = Rate(hz, name=f"session-sampler-{session.id}")
    sent: dict[str, float] = {}
    while True:
        store = session.store
        if store is None:
            sent.clear()
        else:
            state, action = store.read(STATE), store.read(ACTION)
            joints_at = max((sample.timestamp for sample in (state, action) if sample is not None), default=None)
            frames = {
                key: frame
                for key in store.spec.observations
                if (frame := store.read_observation(key)) is not None and frame.timestamp != sent.get(key)
            }
            if frames or (joints_at is not None and joints_at != sent.get(STATE)):
                jpegs = await asyncio.gather(
                    *(asyncio.to_thread(encode_jpeg_rgb, frame.value) for frame in frames.values())
                )
                header = ObservationMessage(
                    timestamp=max([joints_at or 0.0, *(frame.timestamp for frame in frames.values())]),
                    state=_named(state),
                    action=_named(action),
                    frames=[
                        ObservationFrame(key=key, timestamp=frame.timestamp, size=len(jpeg))
                        for (key, frame), jpeg in zip(frames.items(), jpegs, strict=True)
                    ],
                )
                await websocket.send_bytes(observation_message(header, jpegs))
                if joints_at is not None:
                    sent[STATE] = joints_at
                sent.update((key, frame.timestamp) for key, frame in frames.items())
        await rate.sleep()


def observation_message(header: ObservationMessage, jpegs: list[bytes]) -> bytes:
    """``[u32 big-endian header length][header JSON][jpeg bytes, in the order of header.frames]``."""
    encoded = header.model_dump_json().encode()
    return len(encoded).to_bytes(4, "big") + encoded + b"".join(jpegs)

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
from typing import TYPE_CHECKING, Annotated, Any
from uuid import UUID  # noqa: TC003  # FastAPI evaluates websocket annotations at runtime

from fastapi import APIRouter, Depends, WebSocket, status
from fastapi.responses import Response
from fastapi.websockets import WebSocketDisconnect
from loguru import logger

from api.dependencies import EnvironmentServiceDep, RobotClientFactoryDep, get_environment_id, get_project_id
from exceptions import BaseException as AppBaseException
from runtimev2.session import RuntimeSession

if TYPE_CHECKING:
    from runtimev2.store import FeatureStore
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
            "teleoperating": loaded.teleoperating if loaded else False,
            "features": loaded.features if loaded else 0,
        },
    }


def _error_message(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, AppBaseException):
        return {"event": "error", "message": exc.message, "error_code": exc.error_code}
    return {"event": "error", "message": str(exc), "error_code": "runtime_session_failed"}


def _observation_message(store: FeatureStore) -> dict[str, Any]:
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


async def _handle_incoming(
    websocket: WebSocket,
    session: RuntimeSession,
    environment_service: EnvironmentService,
    project_id: UUID,
) -> None:
    """Translate client messages into calls on the session.

    Returns on an explicit ``disconnect``, or when the socket closes. A command
    that fails is reported and the session carries on -- asking to load an
    environment whose robot is unplugged should not drop the connection.
    """
    try:
        while True:
            message = await websocket.receive_json("text")
            event = message.get("event")
            if event == "disconnect":
                return
            try:
                await _apply(websocket, session, environment_service, project_id, message)
            except Exception as exc:
                logger.warning("runtimev2 command {} failed: {}", event, exc)
                await websocket.send_json(_error_message(exc))
    except WebSocketDisconnect:
        logger.debug("runtimev2 websocket closed; ending the session")


async def _apply(
    websocket: WebSocket,
    session: RuntimeSession,
    environment_service: EnvironmentService,
    project_id: UUID,
    message: dict[str, Any],
) -> None:
    """Run one client command, then report the session's new state."""
    event = message.get("event")
    if event == "load_environment":
        environment = await environment_service.get_environment_by_id(
            project_id, get_environment_id(message["environment_id"])
        )
        await session.load(environment)
    elif event == "unload_environment":
        await session.unload()
    elif event == "set_teleoperating":
        session.require_environment().set_teleoperating(bool(message.get("enabled")))
    else:
        logger.debug("Ignoring unknown runtimev2 event {}", event)
        return
    await websocket.send_json(_state_message(session))


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
async def runtimev2_websocket(
    project_id: Annotated[UUID, Depends(get_project_id)],
    environment_service: EnvironmentServiceDep,
    robot_client_factory: RobotClientFactoryDep,
    websocket: WebSocket,
) -> None:
    """Run one environment for as long as this websocket is open.

    Accepts ``{"event": "load_environment", "environment_id": "..."}``,
    ``{"event": "unload_environment"}``,
    ``{"event": "set_teleoperating", "enabled": true}`` and
    ``{"event": "disconnect"}``.

    The session opens empty. Nothing is connected until an environment is
    loaded, and one can be swapped for another without reconnecting.
    """
    await websocket.accept()
    try:
        async with RuntimeSession(robot_client_factory) as session:
            await websocket.send_json(_state_message(session))
            incoming = asyncio.create_task(_handle_incoming(websocket, session, environment_service, project_id))
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

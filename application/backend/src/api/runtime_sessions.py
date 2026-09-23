# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""List and stop the runtime sessions running on this host.

Deliberately not project-scoped. The lock directory is host-wide, a session's
identity carries no project, and a session whose robot row has been deleted has
no project to be listed under.
"""

import asyncio
from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, Path, WebSocket, WebSocketDisconnect, status
from fastapi.responses import Response
from loguru import logger

from api.dependencies import get_event_processor_ws, get_runtime_session_service
from exceptions import BaseException as AppBaseException
from runtime.transport.ids import validate_session_name
from schemas.runtime_session import RuntimeSessionCount, RuntimeSessionInfo
from services.event_processor import EventProcessor, EventType
from services.runtime_session_service import RuntimeSessionService

router = APIRouter(prefix="/api/runtime", tags=["Runtime Sessions"])

RuntimeSessionServiceDep = Annotated[RuntimeSessionService, Depends(get_runtime_session_service)]
SessionName = Annotated[str, Path(description="Runtime session name, `rt-<follower robot id>`.")]


@router.get("/sessions")
async def list_runtime_sessions(service: RuntimeSessionServiceDep) -> list[RuntimeSessionInfo]:
    """List the runtime sessions holding a lock on this host."""
    return await service.list_sessions()


@router.get("/sessions/count")
async def count_runtime_sessions(service: RuntimeSessionServiceDep) -> RuntimeSessionCount:
    """Count the runtime sessions holding a lock on this host.

    Answered from the lock directory alone. Kept for the initial fetch; the
    always-mounted footer gets live updates from `/sessions/ws` instead of
    polling this.
    """
    return RuntimeSessionCount(count=service.count())


@router.get("/sessions/ws", tags=["WebSocket"], summary="Runtime session count updates (WebSocket)", status_code=426)
async def runtime_session_count_websocket_openapi() -> Response:
    """This endpoint requires a WebSocket connection. Use `wss://` to connect."""
    return Response(status_code=426)


@router.websocket("/sessions/ws")
async def runtime_session_count_websocket(
    websocket: WebSocket,
    service: RuntimeSessionServiceDep,
    event_processor: Annotated[EventProcessor, Depends(get_event_processor_ws)],
) -> None:
    """Push the runtime session count as it changes.

    A single host-side watcher task (`RuntimeSessionService.watch_count`)
    polls the lock directory and emits on this event; every connected client
    just receives the resulting pushes instead of polling itself.
    """
    await websocket.accept()

    async def send_count(event: EventType, payload: RuntimeSessionCount) -> None:
        await websocket.send_json({"event": event, "data": payload.model_dump(mode="json")})

    event_processor.subscribe([EventType.RUNTIME_SESSION_COUNT_UPDATE], send_count)
    await send_count(
        EventType.RUNTIME_SESSION_COUNT_UPDATE, RuntimeSessionCount(count=await asyncio.to_thread(service.count))
    )

    try:
        while True:
            await websocket.receive_json("text")
    except WebSocketDisconnect:
        pass
    event_processor.unsubscribe([EventType.RUNTIME_SESSION_COUNT_UPDATE], send_count)


@router.post("/sessions/{session_name}/stop", status_code=status.HTTP_204_NO_CONTENT)
async def stop_runtime_session(session_name: SessionName, service: RuntimeSessionServiceDep) -> None:
    """Terminate a runtime session, releasing its robot and cameras.

    Graceful: the worker takes SIGTERM through the same teardown the idle timeout
    uses, so devices are released and any recording is finalized. It escalates to
    SIGKILL only if that does not land.
    """
    try:
        # Reuses the transport's own validator rather than restating the pattern
        # as a route regex, so the two cannot drift. This is the only thing
        # standing between a path parameter and a signalled pid.
        name = validate_session_name(session_name)
    except ValueError as exc:
        raise AppBaseException(
            message=f"{session_name!r} is not a runtime session name.",
            error_code="invalid_runtime_session_name",
            http_status=HTTPStatus.UNPROCESSABLE_ENTITY,
        ) from exc

    if not await service.stop(name):
        logger.error("Runtime session {} still holds its lock after a stop", name)
        raise AppBaseException(
            message=f"Runtime session {name} did not stop.",
            error_code="runtime_session_stop_failed",
            http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )

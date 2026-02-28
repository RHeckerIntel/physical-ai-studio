# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Log streaming API endpoints.

Provides endpoints to discover available log sources and to stream log file
contents in real-time via Server-Sent Events.
"""

import asyncio
import os
from collections.abc import AsyncGenerator

import anyio
from fastapi import APIRouter, HTTPException, status
from loguru import logger
from sse_starlette import EventSourceResponse, ServerSentEvent

from core.logging.log_config import LogConfig
from core.logging.utils import get_job_logs_path, _validate_job_id
from schemas.logs import LogSource
from settings import get_settings

router = APIRouter(prefix="/api/logs", tags=["Logs"])

# Static mapping from source id to (display name, log filename).
# The log filename is relative to settings.log_dir.
_WORKER_SOURCES: dict[str, tuple[str, str]] = {
    "app": ("Application", "app.log"),
    "training": ("Training", "training.log"),
    "import_export": ("Import / Export", "import_export.log"),
    "inference": ("Inference", "inference.log"),
    "teleoperate": ("Teleoperate", "teleoperate.log"),
}


def _get_worker_log_path(source_id: str) -> str | None:
    """Resolve a worker source id to an absolute log file path."""
    entry = _WORKER_SOURCES.get(source_id)
    if entry is None:
        return None
    settings = get_settings()
    return os.path.join(settings.log_dir, entry[1])


def _discover_job_sources() -> list[LogSource]:
    """List per-job log files on disk and return them as LogSource entries."""
    settings = get_settings()
    jobs_dir = os.path.join(settings.log_dir, "jobs")
    sources: list[LogSource] = []
    if not os.path.isdir(jobs_dir):
        return sources
    for filename in sorted(os.listdir(jobs_dir)):
        if not filename.endswith(".log"):
            continue
        job_id = filename.removesuffix(".log")
        sources.append(LogSource(id=f"job-{job_id}", name=f"Job: {job_id}", type="job"))
    return sources


def _resolve_source_path(source_id: str) -> str | None:
    """Resolve any source id (worker or job) to an absolute log file path.

    Returns None if the source id is not recognized.
    """
    # Worker source
    path = _get_worker_log_path(source_id)
    if path is not None:
        return path

    # Job source (format: "job-<uuid>")
    if source_id.startswith("job-"):
        job_id = source_id[4:]
        try:
            _validate_job_id(job_id)
        except ValueError:
            return None
        return get_job_logs_path(job_id)

    return None


async def _tail_log_file(path: str) -> AsyncGenerator[ServerSentEvent]:
    """Async generator that live-tails a log file and yields SSE events.

    Reads existing content first, then polls for new lines every 500ms.
    The generator runs until the client disconnects (the SSE connection is
    closed), which causes the framework to cancel this coroutine via
    GeneratorExit / CancelledError.
    """
    try:
        async with await anyio.open_file(path, encoding="utf-8") as f:
            while True:
                line = await f.readline()
                if not line:
                    await asyncio.sleep(0.5)
                    continue
                yield ServerSentEvent(data=line.rstrip())
    except asyncio.CancelledError:
        logger.debug(f"SSE log stream cancelled for {path}")
    except GeneratorExit:
        logger.debug(f"SSE log stream closed for {path}")


@router.get("/sources")
async def get_log_sources() -> list[LogSource]:
    """Return all available log sources (worker logs + per-job logs)."""
    sources: list[LogSource] = []
    for source_id, (name, _filename) in _WORKER_SOURCES.items():
        sources.append(LogSource(id=source_id, name=name, type="worker"))
    sources.extend(_discover_job_sources())
    return sources


@router.get("/{source_id}/stream")
async def stream_logs(source_id: str) -> EventSourceResponse:
    """Stream log lines from the given source via Server-Sent Events.

    The connection stays open and new lines are pushed as they are written
    to the log file. Close the connection from the client side to stop.
    """
    path = _resolve_source_path(source_id)
    if path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown log source: {source_id}",
        )
    if not await anyio.Path(path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Log file not found for source: {source_id}",
        )
    return EventSourceResponse(_tail_log_file(path))

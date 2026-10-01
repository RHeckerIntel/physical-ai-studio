"""Parent-side handle for one runtime session worker process."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import queue
import threading
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from exceptions import BaseException as AppBaseException
from runtime.config_builder import runtime_camera_keys
from runtime.contract import QueueEventSink, StateEvent
from runtime.session import RECORDING_TEARDOWN_TIMEOUT_S
from runtime.worker import RuntimeSessionWorker, WorkerFatal
from workers.base import ManagedLifecycle

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence
    from multiprocessing.synchronize import Event as EventClass
    from typing import Self
    from uuid import UUID

    from runtime.contract import Command, RuntimeEvent, StateData
    from runtime.registry import RuntimeSessionRegistry
    from services.camera_claims import CameraClaim, CameraClaimRegistry

# Teardown may copy a recording cache back to the dataset; give it that long
# before escalating to SIGTERM.
STOP_TIMEOUT_S = RECORDING_TEARDOWN_TIMEOUT_S + 5.0
_EVENT_QUEUE_SIZE = 256
_READY_POLL_S = 0.02

SessionStatus = Literal["starting", "running", "stopped", "error"]


class RuntimeProcessError(AppBaseException):
    def __init__(self, message: str, error_code: str = "robot_connection_failed") -> None:
        super().__init__(message=message, error_code=error_code, http_status=500)


class RuntimeSessionHandle(ManagedLifecycle["RuntimeSessionHandle"]):
    """Start, talk to and stop one RuntimeSessionWorker.

    The session lives exactly as long as the ``async with`` that opened it:
    there is no reattach. It holds three things -- the follower's slot in the
    session registry, its cameras' settings claims, and the worker process --
    and :meth:`lifecycle` is where it says so, so no caller has to know the
    list or the order to give them back in.
    """

    def __init__(  # noqa: PLR0913 -- a session's devices and its two registries
        self,
        session_name: str,
        *,
        follower_id: UUID,
        document: dict[str, Any],
        follower_name: str | None,
        leader_name: str | None,
        stop_event: EventClass,
        sessions: RuntimeSessionRegistry,
        claims: CameraClaimRegistry,
        camera_claims: Sequence[CameraClaim] = (),
    ) -> None:
        self.session_name = session_name
        self.follower_id = follower_id
        self.follower_name = follower_name
        self.leader_name = leader_name
        self.camera_keys = runtime_camera_keys(document)
        self._sessions = sessions
        self._claims = claims
        self._camera_claims = camera_claims
        self.started_at = datetime.now(UTC)
        self.state: StateData | None = None
        self.error: RuntimeProcessError | None = None
        self._command_queue: mp.Queue = mp.Queue()
        self._event_queue: mp.Queue = mp.Queue(maxsize=_EVENT_QUEUE_SIZE)
        self._worker = RuntimeSessionWorker(
            document=document,
            follower_name=follower_name,
            leader_name=leader_name,
            stop_event=stop_event,
            command_queue=self._command_queue,
            event_queue=self._event_queue,
        )
        self._events = QueueEventSink()
        self._ready = threading.Event()
        self._stopping = threading.Event()
        self._stopped = threading.Event()
        # Serializes start against stop so a stop during spawn cannot orphan the child.
        self._lifecycle_lock = threading.Lock()
        self._drain_lock = threading.Lock()

    @property
    def pid(self) -> int | None:
        return self._worker.pid

    @property
    def stopping(self) -> bool:
        """Whether a stop has been requested; the slot is on its way to being freed."""
        return self._stopping.is_set()

    @property
    def status(self) -> SessionStatus:
        if self.error is not None:
            return "error"
        if self._stopping.is_set():
            return "stopped"
        if self._ready.is_set():
            return "running"
        return "starting"

    def is_alive(self) -> bool:
        return self._worker.is_alive()

    def start(self) -> None:
        """Spawn the worker. Blocking: a spawn re-imports the backend."""
        with self._lifecycle_lock:
            if self._stopping.is_set():
                raise RuntimeProcessError("Runtime session was stopped before it started")
            self._worker.start()

    def wait_until_ready(self) -> None:
        """Block until the robot reports connected, the worker fails, or a stop is requested."""
        while True:
            self._drain()
            if self._ready.is_set():
                return
            if self.error is not None:
                raise self.error
            if self._stopping.is_set():
                raise RuntimeProcessError("Runtime session was stopped before becoming ready")
            if not self.is_alive():
                # The child flushes its queue before exiting; read what it left.
                self._drain()
                if self.error is not None:
                    raise self.error
                raise RuntimeProcessError("Runtime session stopped before becoming ready")
            time.sleep(_READY_POLL_S)

    def apply(self, command: Command) -> None:
        """Send a command. Acked requests answer with an ``AckEvent`` on the event stream."""
        if self._stopping.is_set():
            return
        self._command_queue.put(command)

    def get_nowait(self) -> RuntimeEvent:
        self._drain()
        return self._events.get_nowait()

    @asynccontextmanager
    async def lifecycle(self) -> AsyncGenerator[Self]:
        """Claim the follower and its cameras, spawn the worker, give all three back.

        Order is the point. The worker is entered last so it stops first: the
        follower must not look free while the process driving it is still
        finalizing a recording and de-energizing the arm, or the next session
        claims a robot whose serial port is still open.
        """
        async with self._sessions.hold(self):
            with self._claims.hold(self._camera_claims):
                try:
                    await asyncio.to_thread(self.start)
                except BaseException:
                    await asyncio.to_thread(self.stop)
                    raise
                try:
                    yield self
                finally:
                    await asyncio.to_thread(self.stop)

    def stop(self) -> None:
        """Stop the worker through its stop event and wait for teardown. Blocking, idempotent."""
        self._stopping.set()
        with self._lifecycle_lock:
            if self._stopped.is_set():
                return
            try:
                self._stop_worker()
            finally:
                # Nothing reads commands any more; do not let a buffered put
                # block this process at exit.
                self._command_queue.cancel_join_thread()
                self._command_queue.close()
                self._stopped.set()

    def _stop_worker(self) -> None:
        # ``on_poll`` keeps the event queue draining while we wait: a child cannot
        # exit while its queue feeder is blocked on a full pipe.
        self._worker.stop(STOP_TIMEOUT_S, on_poll=self._drain)

    def _drain(self) -> None:
        with self._drain_lock:
            while True:
                try:
                    item = self._event_queue.get_nowait()
                except (queue.Empty, OSError, ValueError):
                    return
                self._accept(item)

    def _accept(self, item: RuntimeEvent | WorkerFatal) -> None:
        if isinstance(item, WorkerFatal):
            self.error = RuntimeProcessError(item.message, item.error_code)
            return
        if isinstance(item, StateEvent):
            self.state = item.data
            if item.data.connected:
                self._ready.set()
        self._events.emit(item)

"""A runtime session: what one websocket has loaded, and the workers running it.

The session lives in the API process and belongs to the websocket that opened
it. It owns everything it loads; closing it releases all of it, in reverse.
For now it loads one environment (see ``runtime_session/environment.py``).
The session adds what is about the session itself: its claims on the devices,
watching the workers, and telling its socket about changes.

See the "RuntimeSession API" section of docs/feature_plans/runtime-inhousing-plan.md.
"""

from __future__ import annotations

import asyncio
import datetime
from contextlib import AsyncExitStack, contextmanager
from typing import TYPE_CHECKING
from uuid import uuid4

import anyio
from loguru import logger

from exceptions import RuntimeSessionCommandError
from runtime_session.environment import connect_environment, describe_environment
from schemas.runtime_session import ErrorMessage, RuntimeSessionInfo, SessionEnvironment, SessionState, StateMessage
from services.camera_claims import CameraClaim, settings_from_camera

if TYPE_CHECKING:
    from collections.abc import Iterator
    from uuid import UUID

    from physicalai.robot import Robot

    from robots.robot_client_factory import RobotClientFactory
    from runtime_session.environment import ConnectedEnvironment, EnvironmentDescription
    from runtime_session.registry import RuntimeSessionRegistry
    from runtime_session.store import Store
    from schemas.environment import EnvironmentWithRelations
    from services.camera_claims import CameraClaimRegistry

_WATCH_INTERVAL_S = 0.25

type SessionEvent = StateMessage | ErrorMessage


class RuntimeSession:
    def __init__(
        self,
        *,
        project_id: UUID,
        project_name: str,
        registry: RuntimeSessionRegistry,
        camera_claims: CameraClaimRegistry,
        robot_factory: RobotClientFactory,
    ) -> None:
        self.id = uuid4().hex
        self.project_id = project_id
        self.project_name = project_name
        self.started_at = datetime.datetime.now(tz=datetime.UTC)
        self.events: asyncio.Queue[SessionEvent] = asyncio.Queue()
        """State changes and errors, for the owning websocket."""
        self.closed = asyncio.Event()
        self._registry = registry
        self._camera_claims = camera_claims
        self._robot_factory = robot_factory
        self._lock = asyncio.Lock()
        self._environment: SessionEnvironment | None = None
        self._connected: ConnectedEnvironment | None = None
        self._environment_stack: AsyncExitStack | None = None
        self._watch_task: asyncio.Task[None] | None = None
        self._unload_task: asyncio.Task[None] | None = None

    @property
    def label(self) -> str:
        """How errors name this session to a user."""
        if self._environment is not None:
            return f"the runtime session for {self._environment.name!r} (project {self.project_name!r})"
        return f"a runtime session in project {self.project_name!r}"

    @property
    def environment(self) -> SessionEnvironment | None:
        return self._environment

    @property
    def store(self) -> Store | None:
        return self._connected.store if self._connected is not None else None

    @property
    def leader(self) -> Robot | None:
        """The environment's leader, connected, for controls to borrow."""
        return self._connected.leader if self._connected is not None else None

    def state(self) -> SessionState:
        return SessionState(session_id=self.id, environment=self._environment)

    def info(self) -> RuntimeSessionInfo:
        environment = self._environment
        return RuntimeSessionInfo(
            session_id=self.id,
            project_id=self.project_id,
            project_name=self.project_name,
            started_at=self.started_at,
            environment_name=environment.name if environment else None,
            robot_ids=[robot for robot in (environment.robot_id, environment.leader_id) if robot]
            if environment
            else [],
            camera_ids=[camera.camera_id for camera in environment.cameras] if environment else [],
        )

    async def load_environment(self, environment: EnvironmentWithRelations) -> None:
        """Connect the environment's robot, leader and cameras, or fail with nothing held.

        Raises:
            RuntimeSessionCommandError: An environment is already loaded.
            DeviceInUseError: Another session holds one of its robots.
            CameraSettingsConflictError: Another session uses a camera at other settings.
            WorkerStartupError: A device failed to connect.
        """
        async with self._lock:
            if self._environment_stack is not None:
                raise RuntimeSessionCommandError(
                    "An environment is already loaded; unload it first.", error_code="environment_already_loaded"
                )
            description = await describe_environment(environment, self._robot_factory)
            async with AsyncExitStack() as stack:
                stack.enter_context(self._claim(description))
                connected = await stack.enter_async_context(connect_environment(description, self.id))
                self._watch_task = asyncio.create_task(self._watch(connected))
                self._connected = connected
                self._environment = description.session_environment()
                self._environment_stack = stack.pop_all()
            logger.info("Session {} loaded environment {!r}", self.id, environment.name)
        self._emit(StateMessage(data=self.state()))

    @contextmanager
    def _claim(self, description: EnvironmentDescription) -> Iterator[None]:
        """Claim the robots and pin the camera settings for this session, so no form or other session takes them.

        Raises:
            DeviceInUseError: Another session holds one of the robots.
            CameraSettingsConflictError: Another session uses a camera at other settings.
        """
        claims = []
        for camera in description.cameras.values():
            if camera.fingerprint is None:
                raise ValueError(f"Camera {camera.name!r} must be reselected")
            claims.append(
                CameraClaim(
                    fingerprint=camera.fingerprint,
                    settings=settings_from_camera(camera),
                    holder=self.id,
                    project_id=self.project_id,
                    project_name=self.project_name,
                )
            )
        self._registry.claim_robots(self, dict(description.robot_names))
        try:
            generation = self._camera_claims.claim(claims)
            try:
                yield
            finally:
                self._camera_claims.release(self.id, generation=generation)
        finally:
            self._registry.release_robots(self)

    async def unload_environment(self) -> None:
        """Disconnect the environment. A no-op when none is loaded."""
        async with self._lock:
            stack, self._environment_stack = self._environment_stack, None
            if stack is None:
                return
            watch, self._watch_task = self._watch_task, None
            if watch is not None and watch is not asyncio.current_task():
                watch.cancel()
            # Cleared before the teardown awaits, so nothing samples a store being unlinked.
            self._connected = None
            name = self._environment.name if self._environment else "?"
            self._environment = None
            await stack.aclose()
            logger.info("Session {} unloaded environment {!r}", self.id, name)
        self._emit(StateMessage(data=self.state()))

    async def close(self, reason: str) -> None:
        """Release everything. ``reason`` other than ``socket_closed`` is sent to the socket as a final error."""
        if self.closed.is_set():
            return
        self.closed.set()
        with anyio.CancelScope(shield=True):
            try:
                await self.unload_environment()
            finally:
                self._registry.remove(self)
                logger.info("Session {} closed ({})", self.id, reason)
        if reason != "socket_closed":
            self._emit(
                ErrorMessage(
                    component="session",
                    message="The runtime session was closed.",
                    error_code="session_closed",
                    session_alive=False,
                )
            )

    async def _watch(self, connected: ConnectedEnvironment) -> None:
        """Unload the environment when one of its workers dies; the session itself survives."""
        while True:
            await asyncio.sleep(_WATCH_INTERVAL_S)
            for worker in connected.workers:
                if not worker.is_alive():
                    logger.error("Session {}: worker {} stopped: {}", self.id, worker.name, worker.failure)
                    self._emit(
                        ErrorMessage(
                            component=worker.name,
                            message=worker.failure or f"{worker.name} stopped",
                            error_code="worker_failed",
                            session_alive=True,
                        )
                    )
                    self._unload_task = asyncio.create_task(self.unload_environment())
                    return

    def _emit(self, event: SessionEvent) -> None:
        self.events.put_nowait(event)
        if event.type == "state":
            self._registry.changed()

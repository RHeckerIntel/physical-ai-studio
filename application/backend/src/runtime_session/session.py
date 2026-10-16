"""A runtime session: what one websocket has loaded, and the workers running it.

The session lives in the API process and belongs to the websocket that opened
it: the socket holds it with ``async with``, and leaving that block releases
everything it loaded, in reverse. While it runs, its optional parts are loaded
and unloaded independently, each in its own :class:`Slot`: for now the
environment (see ``runtime_session/environment.py``) and the control driving
the follower. The session adds what is about the session itself: its claims on
the devices, watching the workers, and telling its socket about changes.

See the "RuntimeSession API" section of docs/feature_plans/runtime-inhousing-plan.md.
"""

from __future__ import annotations

import asyncio
import datetime
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self
from uuid import uuid4

import anyio
from loguru import logger

from exceptions import RuntimeSessionCommandError
from runtime_session.controls import build_control
from runtime_session.environment import connect_environment, describe_environment
from runtime_session.slot import Slot
from runtime_session.store import ACTION, STATE
from schemas.runtime_session import ErrorMessage, RuntimeSessionInfo, SessionEnvironment, SessionState, StateMessage
from services.camera_claims import CameraClaim, settings_from_camera

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Coroutine, Iterator
    from uuid import UUID

    from physicalai.robot import Robot

    from robots.robot_client_factory import RobotClientFactory
    from runtime_session.controls import Control
    from runtime_session.environment import ConnectedEnvironment, EnvironmentDescription
    from runtime_session.registry import RuntimeSessionRegistry
    from runtime_session.store import Store
    from schemas.environment import EnvironmentWithRelations
    from schemas.runtime_session import ControlConfig
    from services.camera_claims import CameraClaimRegistry

_WATCH_INTERVAL_S = 0.25

type SessionEvent = StateMessage | ErrorMessage


@dataclass(frozen=True, slots=True)
class LoadedEnvironment:
    connected: ConnectedEnvironment
    state: SessionEnvironment


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
        self._environment: Slot[LoadedEnvironment] = Slot("environment")
        self._control: Slot[Control] = Slot("control")
        """Uses the environment's store and leader, so it is unloaded before the environment."""
        self._background: set[asyncio.Task[None]] = set()

    async def __aenter__(self) -> Self:
        self._registry.add(self)
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self._teardown("socket closed")

    async def stop(self) -> None:
        """Close the session from outside its socket; the socket gets ``session_closed`` and closes."""
        await self._teardown("stopped")
        self._emit(
            ErrorMessage(
                component="session",
                message="The runtime session was closed.",
                error_code="session_closed",
                session_alive=False,
            )
        )

    async def _teardown(self, reason: str) -> None:
        """Unload everything and unregister. Runs once, whichever of the socket and a stop gets here first."""
        if self.closed.is_set():
            return
        self.closed.set()
        with anyio.CancelScope(shield=True):
            try:
                await self.unload_environment()
            finally:
                self._registry.remove(self)
                logger.info("Session {} closed ({})", self.id, reason)

    @property
    def label(self) -> str:
        """How errors name this session to a user."""
        if (environment := self.environment) is not None:
            return f"the runtime session for {environment.name!r} (project {self.project_name!r})"
        return f"a runtime session in project {self.project_name!r}"

    @property
    def environment(self) -> SessionEnvironment | None:
        loaded = self._environment.value
        return loaded.state if loaded is not None else None

    @property
    def store(self) -> Store | None:
        loaded = self._environment.value
        return loaded.connected.store if loaded is not None else None

    @property
    def leader(self) -> Robot | None:
        """The environment's leader, connected, for controls to borrow."""
        loaded = self._environment.value
        return loaded.connected.leader if loaded is not None else None

    def state(self) -> SessionState:
        control = self._control.value
        return SessionState(
            session_id=self.id, environment=self.environment, control=control.config if control else None
        )

    def info(self) -> RuntimeSessionInfo:
        environment = self.environment
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
            RuntimeSessionCommandError: The session is closed, or an environment is already loaded.
            DeviceInUseError: Another session holds one of its robots.
            CameraSettingsConflictError: Another session uses a camera at other settings.
            WorkerStartupError: A device failed to connect.
        """
        async with self._lock:
            if self.closed.is_set():
                raise RuntimeSessionCommandError("The session is closed.", error_code="session_closed")
            await self._environment.load(self._loaded_environment(environment))
            logger.info("Session {} loaded environment {!r}", self.id, environment.name)
        self._emit(StateMessage(data=self.state()))

    @asynccontextmanager
    async def _loaded_environment(self, environment: EnvironmentWithRelations) -> AsyncIterator[LoadedEnvironment]:
        """Claim, connect and watch the environment for as long as it is loaded."""
        description = await describe_environment(environment, self._robot_factory)
        with self._claim(description):
            async with connect_environment(description, self.id) as connected:
                watch = asyncio.create_task(self._watch(connected))
                try:
                    yield LoadedEnvironment(connected=connected, state=description.session_environment())
                finally:
                    watch.cancel()

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

    async def set_control(self, config: ControlConfig | None) -> None:
        """Stop the running control, hold the follower where it is, then start ``config``.

        Raises:
            RuntimeSessionCommandError: No environment is loaded.
            ValueError: The environment cannot run ``config`` (no leader, unknown joints).
            WorkerStartupError: The control failed to start; the follower holds.
        """
        async with self._lock:
            loaded = self._environment.require("Load an environment first.")
            try:
                await self._switch_control(loaded.connected, config)
            finally:
                self._emit(StateMessage(data=self.state()))

    async def _switch_control(self, connected: ConnectedEnvironment, config: ControlConfig | None) -> None:
        """Hold the lock. One control at a time, by construction: the old one is joined before the next starts."""
        await self._control.unload()
        # Hold: the follower stays where it is now, not where the old control last pointed it.
        if (state := connected.store.read(STATE)) is not None:
            connected.store.write(ACTION, state.value, timestamp=time.monotonic())
        if config is not None:
            await self._control.load(build_control(config, connected.store, connected.leader))
            logger.info("Session {} switched to {} control", self.id, config.type)

    async def unload_environment(self) -> None:
        """Disconnect the environment. A no-op when none is loaded."""
        async with self._lock:
            if (environment := self.environment) is None:
                return
            await self._control.unload()
            await self._environment.unload()
            logger.info("Session {} unloaded environment {!r}", self.id, environment.name)
        self._emit(StateMessage(data=self.state()))

    async def _watch(self, connected: ConnectedEnvironment) -> None:
        """Unload the environment when one of its workers dies, and hold when the control dies.

        The session itself survives either.
        """
        reported: Control | None = None
        while True:
            await asyncio.sleep(_WATCH_INTERVAL_S)
            if (control := self._control.value) is not None and control is not reported and not control.is_alive():
                reported = control
                logger.error("Session {}: control {} stopped: {}", self.id, control.name, control.failure)
                self._emit(
                    ErrorMessage(
                        component=control.name,
                        message=control.failure or f"{control.name} stopped",
                        error_code="control_failed",
                        session_alive=True,
                    )
                )
                self._run_in_background(self._hold_after(control))
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
                    self._run_in_background(self.unload_environment())
                    return

    async def _hold_after(self, control: Control) -> None:
        async with self._lock:
            loaded = self._environment.value
            if loaded is None or self._control.value is not control:
                return  # switched or unloaded meanwhile
            await self._switch_control(loaded.connected, None)
        self._emit(StateMessage(data=self.state()))

    def _run_in_background(self, coroutine: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coroutine)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    def _emit(self, event: SessionEvent) -> None:
        self.events.put_nowait(event)
        if event.type == "state":
            self._registry.changed()

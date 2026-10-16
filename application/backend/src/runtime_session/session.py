"""A runtime session: what one websocket has loaded, and the workers running it.

The session lives in the API process and belongs to the websocket that opened
it: the socket holds it with ``async with``, and leaving that block releases
everything it loaded, in reverse. Like the workers, it is a part on the bus:
while it runs it serves its own commands (loading, unloading and switching
control, see ``command_key``) and publishes its state and errors on
``evt/session``. The socket only forwards between the two.

Its optional parts are loaded and unloaded independently, each in its own
:class:`Slot`: the environment (see ``runtime_session/environment.py``), the
control driving the follower, and the dataset being recorded. The session adds
what is about the session itself: its claims on the devices and watching the
workers.

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

from exceptions import BaseException as AppBaseException
from exceptions import RuntimeSessionCommandError
from runtime_session.bus import Bus, NoResponderError, Subscription, command_key
from runtime_session.controls import build_control
from runtime_session.dataset import dataset_fps, dataset_spec
from runtime_session.environment import connect_environment, describe_environment
from runtime_session.slot import Slot
from runtime_session.store import ACTION, STATE, FeatureSpecMismatchError
from schemas.runtime_session import (
    CommandDone,
    ErrorMessage,
    LoadDatasetCommand,
    LoadEnvironmentCommand,
    RuntimeSessionInfo,
    SessionDataset,
    SessionEnvironment,
    SessionState,
    SetControlCommand,
    StateMessage,
    UnloadDatasetCommand,
    UnloadEnvironmentCommand,
)
from services.camera_claims import CameraClaim, settings_from_camera
from workers import dataset_worker as dataset_rpc
from workers.base import WorkerStartupError
from workers.dataset_worker import DatasetWorker, RecordingStatus

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator
    from uuid import UUID

    from physicalai.robot import Robot
    from pydantic import BaseModel

    from robots.robot_client_factory import RobotClientFactory
    from runtime_session.controls import Control
    from runtime_session.environment import ConnectedEnvironment, EnvironmentDescription
    from runtime_session.registry import RuntimeSessionRegistry
    from runtime_session.store import Store
    from schemas.dataset import Dataset
    from schemas.environment import EnvironmentWithRelations
    from schemas.runtime_session import ControlConfig, EnvironmentDraft
    from services.camera_claims import CameraClaimRegistry

_WATCH_INTERVAL_S = 0.25

EVENTS = "evt/session"
"""Where the session publishes its state and errors (``SessionEvent``)."""


_DATASET_READY_S = 10.0
"""How long a started dataset worker has to answer on the bus."""


@dataclass(frozen=True, slots=True)
class Resolvers:
    """Look up what a command names, in the session's project; supplied by whoever opens the session."""

    environment: Callable[[EnvironmentDraft], Awaitable[EnvironmentWithRelations]]
    dataset: Callable[[UUID], Awaitable[Dataset]]


@dataclass(frozen=True, slots=True)
class LoadedEnvironment:
    connected: ConnectedEnvironment
    state: SessionEnvironment
    robot_type: str
    """The follower's type, which a new dataset records."""


@dataclass(slots=True)
class LoadedDataset:
    dataset: Dataset
    fps: int
    status: RecordingStatus
    """What the worker last published on ``evt/dataset``; the worker owns it."""

    def state(self) -> SessionDataset:
        return SessionDataset(
            dataset_id=self.dataset.id,
            name=self.dataset.name,
            fps=self.fps,
            is_recording=self.status.is_recording,
            episodes_recorded=self.status.episodes_recorded,
        )


class RuntimeSession:
    def __init__(
        self,
        *,
        project_id: UUID,
        project_name: str,
        registry: RuntimeSessionRegistry,
        camera_claims: CameraClaimRegistry,
        robot_factory: RobotClientFactory,
        resolve: Resolvers,
    ) -> None:
        self.id = uuid4().hex
        self.project_id = project_id
        self.project_name = project_name
        self.started_at = datetime.datetime.now(tz=datetime.UTC)
        self.closed = asyncio.Event()
        self._resolve = resolve
        self._registry = registry
        self._camera_claims = camera_claims
        self._robot_factory = robot_factory
        self._lock = asyncio.Lock()
        self._environment: Slot[LoadedEnvironment] = Slot("environment")
        self._control: Slot[Control] = Slot("control")
        """Uses the environment's store and leader, so it is unloaded before the environment."""
        self._dataset: Slot[LoadedDataset] = Slot("dataset")
        """Records the environment's store, so it is unloaded before the environment."""
        self._bus: Bus | None = None
        """Hosted for the session's lifetime; its worker processes join it."""
        self._background: set[asyncio.Task[None]] = set()

    async def __aenter__(self) -> Self:
        self._bus = Bus.host(self.id)
        self._serve_commands(self._bus)
        self._registry.add(self)
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self._teardown("socket closed", notify=False)

    async def stop(self) -> None:
        """Close the session from outside its socket; the socket gets ``session_closed`` and closes."""
        await self._teardown("stopped", notify=True)

    async def _teardown(self, reason: str, *, notify: bool) -> None:
        """Unload everything and unregister. Runs once, whichever of the socket and a stop gets here first."""
        if self.closed.is_set():
            return
        self.closed.set()
        with anyio.CancelScope(shield=True):
            try:
                await self.unload_environment()
            finally:
                self._registry.remove(self)
                if notify:
                    self._emit(
                        ErrorMessage(
                            component="session",
                            message="The runtime session was closed.",
                            error_code="session_closed",
                            session_alive=False,
                        )
                    )
                if self._bus is not None:
                    self._bus.close()
                logger.info("Session {} closed ({})", self.id, reason)

    def _serve_commands(self, bus: Bus) -> None:
        """Serve the commands that change what the session holds; workers serve their own."""
        commands: dict[str, tuple[type[BaseModel], Callable[[Any], Awaitable[None]]]] = {
            "load_environment": (LoadEnvironmentCommand, lambda command: self._load_draft(command.environment)),
            "unload_environment": (UnloadEnvironmentCommand, lambda _command: self.unload_environment()),
            "set_control": (SetControlCommand, lambda command: self.set_control(command.control)),
            "load_dataset": (LoadDatasetCommand, lambda command: self._load_dataset_by_id(command.dataset_id)),
            "unload_dataset": (UnloadDatasetCommand, lambda _command: self.unload_dataset()),
        }
        for name, (model, run) in commands.items():
            bus.serve(command_key(name), model, _as_command(run))

    async def _load_draft(self, draft: EnvironmentDraft) -> None:
        await self.load_environment(await self._resolve.environment(draft))

    async def _load_dataset_by_id(self, dataset_id: UUID) -> None:
        await self.load_dataset(await self._resolve.dataset(dataset_id))

    @property
    def label(self) -> str:
        """How errors name this session to a user."""
        if (environment := self.environment) is not None:
            return f"the runtime session for {environment.name!r} (project {self.project_name!r})"
        return f"a runtime session in project {self.project_name!r}"

    @property
    def bus(self) -> Bus:
        """The session's bus: commands go onto it, and its events come off it."""
        if self._bus is None:
            raise RuntimeSessionCommandError("The session is not open.", error_code="session_closed")
        return self._bus

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
        control, dataset = self._control.value, self._dataset.value
        return SessionState(
            session_id=self.id,
            environment=self.environment,
            control=control.config if control else None,
            dataset=dataset.state() if dataset else None,
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
                    yield LoadedEnvironment(
                        connected=connected,
                        state=description.session_environment(),
                        robot_type=environment.robots[0].robot.type,
                    )
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
            await self._dataset.unload()
            await self._environment.unload()
            logger.info("Session {} unloaded environment {!r}", self.id, environment.name)
        self._emit(StateMessage(data=self.state()))

    async def load_dataset(self, dataset: Dataset) -> None:
        """Start recording into ``dataset`` from the environment's store.

        Raises:
            RuntimeSessionCommandError: No environment is loaded, or a dataset already is.
            FeatureSpecMismatchError: The dataset was recorded with joints or cameras this environment lacks.
            WorkerStartupError: The dataset could not be opened for recording.
        """
        async with self._lock:
            environment = self._environment.require("Load an environment first.")
            await self._dataset.load(self._loaded_dataset(dataset, environment))
            logger.info("Session {} records into dataset {!r}", self.id, dataset.name)
        self._emit(StateMessage(data=self.state()))

    @asynccontextmanager
    async def _loaded_dataset(self, dataset: Dataset, environment: LoadedEnvironment) -> AsyncIterator[LoadedDataset]:
        store = environment.connected.store
        if (recorded := dataset_spec(dataset)) is not None:
            store.spec.require_compatible(recorded)
        fps = dataset_fps(dataset)
        worker = DatasetWorker(
            dataset,
            session_id=self.id,
            store_name=store.name,
            spec=store.spec,
            fps=fps,
            robot_type=environment.robot_type,
        )
        # Subscribed before the worker starts, so no status it publishes is missed.
        with self.bus.subscribe(dataset_rpc.EVENTS, RecordingStatus) as statuses:
            async with worker:
                loaded = LoadedDataset(dataset=dataset, fps=fps, status=await self._until_serving())
                tasks = [
                    asyncio.create_task(self._fold_statuses(loaded, statuses)),
                    asyncio.create_task(self._watch_dataset(worker)),
                ]
                try:
                    yield loaded
                finally:
                    for task in tasks:
                        task.cancel()

    async def _until_serving(self) -> RecordingStatus:
        """The worker's first status: its queryables take a moment to reach this process's bus."""
        deadline = time.monotonic() + _DATASET_READY_S
        while True:
            try:
                return await self.bus.request(dataset_rpc.STATUS, dataset_rpc.Empty(), RecordingStatus, timeout=1.0)
            except (NoResponderError, TimeoutError):
                if time.monotonic() > deadline:
                    raise
                await asyncio.sleep(0.05)

    async def _fold_statuses(self, loaded: LoadedDataset, statuses: Subscription[RecordingStatus]) -> None:
        """The worker owns its recording state; show whatever it last announced."""
        while True:
            loaded.status = await statuses.next()
            self._emit(StateMessage(data=self.state()))

    async def unload_dataset(self) -> None:
        """Stop recording and copy what was recorded back into the dataset. A no-op when none is loaded."""
        async with self._lock:
            await self._dataset.unload()
        self._emit(StateMessage(data=self.state()))

    async def _watch_dataset(self, worker: DatasetWorker) -> None:
        """Unload the dataset if its worker dies; what it saved but did not copy back is lost."""
        await worker.wait_stopped()
        logger.error("Session {}: dataset worker stopped: {}", self.id, worker.failure)
        self._emit(
            ErrorMessage(
                component=worker.name,
                message=worker.failure or "The dataset worker stopped",
                error_code="worker_failed",
                session_alive=True,
            )
        )
        self._run_in_background(self.unload_dataset())

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

    def _emit(self, event: StateMessage | ErrorMessage) -> None:
        if self._bus is not None:
            self._bus.publish(EVENTS, event)
        if event.type == "state":
            self._registry.changed()


def _as_command(run: Callable[[Any], Awaitable[None]]) -> Callable[[Any], Awaitable[CommandDone]]:
    """A bus handler for a session command, failing with an error code its requester can show."""

    async def handle(command: Any) -> CommandDone:
        try:
            await run(command)
        except Exception as error:
            raise _command_error(error) from error
        return CommandDone()

    return handle


_ERROR_CODES: tuple[tuple[type[Exception], str], ...] = (
    (FeatureSpecMismatchError, "feature_mismatch"),  # before ValueError, which it is
    (TimeoutError, "command_timeout"),
    (ValueError, "invalid_session_command"),
)


def _command_error(error: Exception) -> Exception:
    if isinstance(error, AppBaseException):
        return error
    if isinstance(error, WorkerStartupError):
        return RuntimeSessionCommandError(f"{error.worker_name}: {error.reason}", error_code="device_connection_failed")
    code = next((code for kind, code in _ERROR_CODES if isinstance(error, kind)), None)
    if code is None:
        logger.opt(exception=error).error("Runtime session command failed")
        code = "internal_error"
    return RuntimeSessionCommandError(str(error) or type(error).__name__, error_code=code)

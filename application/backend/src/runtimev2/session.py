"""A client's runtime session: a dataset, a model, and one loaded environment.

The session is the connection, not the hardware. It holds no robots and no
store of its own -- it loads an environment, which owns those, and can unload it
and load another without the client reconnecting.

A dataset and a model sit *above* the environment. Both are checked against the
:class:`SessionShape`, not against the connected devices, so neither is dropped
when the devices are: unloading and reloading an environment re-attaches them,
and only a shape that no longer fits unloads one. They each carry their own
rate -- a dataset records at the rate its frames are stored at, a model infers
as fast as it can be run, and neither has to match the robots.

Loading an environment is exclusive rather than additive. Two environments
sharing a robot would both hold the same device, so the previous one is unloaded
first and its arms are released before the next one claims anything.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from runtime.callbacks.recording import RecordingState
from runtimev2.control.config import ControlConfig, ModelControlConfig, TeleopControlConfig
from runtimev2.control.model import ModelControl
from runtimev2.control.teleop import TeleopControl
from runtimev2.environment import describe_environment
from runtimev2.inference import LoadedModel, load_model
from runtimev2.leader import open_leaders
from runtimev2.loaded_environment import DEFAULT_ROBOT_HZ, LoadedEnvironment
from runtimev2.recording import LoadedDataset, open_for_recording
from runtimev2.workers.dataset import DatasetWorker
from workers.base import ManagedLifecycle

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path
    from uuid import UUID

    from physicalai.inference import InferenceModel

    from robots.robot_client_factory import RobotClientFactory
    from runtimev2.control.base import ControlAlgorithm
    from runtimev2.loaded_environment import EnvironmentState
    from schemas.environment import EnvironmentWithRelations

# What a brand new dataset records at when the caller does not say. An existing
# one always uses its own rate.
DEFAULT_DATASET_HZ = 30
# What a policy emits at when the caller does not say. Inference itself runs
# every ``chunk_size`` emits, so this is not the rate the GPU sees.
DEFAULT_MODEL_HZ = 30.0

_DATASET_WORKER = "dataset"


@dataclass(frozen=True, slots=True)
class SessionState:
    """What the session looks like from outside.

    ``environment`` is ``None`` when nothing is loaded, which is the state a
    session starts in and returns to after an unload. A dataset can be loaded
    with no environment: it is only checked against one when one arrives.
    """

    loaded: bool
    environment: EnvironmentState | None = None
    dataset_loaded: bool = False
    dataset_id: str | None = None
    dataset_hz: int | None = None
    is_recording: bool = False
    episodes_recorded: int = 0
    task: str | None = None
    model_loaded: bool = False
    model_id: str | None = None
    model_chunk_size: int | None = None
    control: ControlConfig | None = None
    """What the session drives with, rate included, if anything.

    There is no second switch: an arm follows its action features always, and
    they only change while a control is writing them.
    """


class RuntimeSession(ManagedLifecycle["RuntimeSession"]):
    """One client connection, holding at most one loaded environment.

    Opened with ``async with``; leaving unloads whatever is still loaded.
    """

    def __init__(self, factory: RobotClientFactory, *, robot_hz: float = DEFAULT_ROBOT_HZ) -> None:
        self._factory = factory
        self._robot_hz = robot_hz
        self._loaded: LoadedEnvironment | None = None
        # The loaded environment's own teardown, kept so it can be closed on its
        # own rather than only when the session ends.
        self._stack: AsyncExitStack | None = None
        # Above the environment: chosen once and re-attached across reloads.
        self._dataset: LoadedDataset | None = None
        self._recording = RecordingState()
        self._task: str | None = None
        # What was asked for, kept so it can be opened once an environment
        # arrives and re-opened against a new shape after a swap.
        self._pending_dataset: tuple[UUID, Path, int] | None = None
        self._policy: InferenceModel | None = None
        self._model: LoadedModel | None = None
        self._pending_model: tuple[UUID, Path, str] | None = None
        # The choice, and the control running it. The control holds a store, so
        # it is rebuilt against each environment -- but it is held here rather
        # than by the environment, because what drives a session is the
        # session's decision and outlives any one set of devices.
        self._control: ControlConfig | None = None
        self._control_worker: ControlAlgorithm | None = None
        self._control_stack: AsyncExitStack | None = None

    @property
    def environment(self) -> LoadedEnvironment | None:
        """The loaded environment, or ``None``."""
        return self._loaded

    def require_environment(self) -> LoadedEnvironment:
        """Return the loaded environment.

        Raises:
            RuntimeError: Nothing is loaded.
        """
        if self._loaded is None:
            raise RuntimeError("No environment is loaded")
        return self._loaded

    def state(self) -> SessionState:
        environment = self._loaded.state() if self._loaded is not None else None
        return SessionState(
            loaded=self._loaded is not None,
            environment=environment,
            dataset_loaded=self._recording.dataset_loaded,
            dataset_id=str(self._dataset.dataset_id) if self._dataset else None,
            dataset_hz=self._dataset.fps if self._dataset else None,
            is_recording=self._recording.is_recording,
            episodes_recorded=self._recording.episodes_recorded,
            task=self._task,
            model_loaded=self._model is not None,
            model_id=str(self._model.model_id) if self._model else None,
            model_chunk_size=self._model.chunk_size if self._model else None,
            control=self._control,
        )

    async def load(self, environment: EnvironmentWithRelations) -> LoadedEnvironment:
        """Load ``environment``, unloading whatever was loaded before.

        A failed load leaves the session empty rather than holding a half-built
        environment: whatever was acquired before the failure is released, and
        the previous environment is already gone by then -- it is unloaded
        first so its robots are free for this one to claim.
        """
        await self.unload()
        stack = AsyncExitStack()
        try:
            # Described here rather than inside the environment: the shape is
            # what a dataset and a model are checked against, so the session
            # needs it whether or not the devices come up.
            shape = await describe_environment(environment, self._factory)
            leaders = await open_leaders(environment, shape, self._factory, stack)
            loaded = await stack.enter_async_context(
                LoadedEnvironment(environment, self._factory, shape, leaders, robot_hz=self._robot_hz)
            )
        except BaseException:
            await stack.aclose()
            raise
        self._loaded, self._stack = loaded, stack
        # The dataset and the model outlive an environment, so both are
        # re-checked and re-attached here rather than dropped by the swap.
        try:
            await self._open_dataset()
        except Exception:
            logger.exception("Dataset does not fit {}; unloading it", loaded.name)
            await self.unload_dataset()
        try:
            await self._open_policy()
        except Exception:
            logger.exception("Model does not fit {}; unloading it", loaded.name)
            await self.unload_policy()
        # Last, because a model becomes the selected control when it opens.
        try:
            await self._open_control()
        except Exception:
            logger.exception("Control does not fit {}; selecting nothing", loaded.name)
            await self.select_control(None)
        return loaded

    async def load_dataset(self, dataset_id: UUID, path: Path, *, hz: int = DEFAULT_DATASET_HZ) -> LoadedDataset:
        """Open a dataset for recording, replacing any already open.

        Checked against the shape of whatever environment is loaded. With none
        loaded the dataset is still accepted and opened lazily -- the check and
        the worker both arrive with the environment, because the row's columns
        come from its devices.

        Raises:
            DatasetIncompatibleError: Its rows do not match the loaded shape.
        """
        await self.unload_dataset()
        self._pending_dataset = (dataset_id, path, hz)
        if self._loaded is not None:
            await self._open_dataset()
        else:
            logger.info("Dataset {} selected; it will open when an environment is loaded", dataset_id)
        return self._dataset or LoadedDataset(dataset_id=dataset_id, path=path, fps=hz)

    async def unload_dataset(self) -> None:
        """Stop recording and close the dataset. Idempotent.

        Finalizes whatever is open: an episode still being recorded is dropped
        rather than half written, and the cache is copied back.
        """
        self._pending_dataset = None
        if self._loaded is not None:
            await self._loaded.detach(_DATASET_WORKER)
        mutation = self._recording.take_mutation()
        if mutation is not None:
            await asyncio.to_thread(mutation.teardown)
        self._dataset = None
        self._task = None

    async def _open_dataset(self) -> None:
        """Open the pending dataset against the loaded environment and attach its worker."""
        if self._pending_dataset is None or self._loaded is None:
            return
        dataset_id, path, hz = self._pending_dataset
        loaded, mutation = await asyncio.to_thread(
            open_for_recording, dataset_id, path, self._loaded.shape, requested_fps=hz
        )
        self._recording.attach_mutation(mutation)
        self._dataset = loaded
        await self._loaded.attach(
            _DATASET_WORKER,
            DatasetWorker(self._loaded.store, self._recording, self._loaded.shape, hz=float(loaded.fps)),
        )

    def start_recording(self, task: str) -> None:
        """Begin an episode.

        Raises:
            RuntimeError: No dataset is open.
        """
        if not self._recording.start(task):
            raise RuntimeError("No dataset is open to record into")
        self._task = task
        self._retask()
        logger.info("Recording started for task {!r}", task)

    def set_task(self, task: str | None) -> None:
        """Set the instruction a policy is conditioned on and a recording stores."""
        self._task = task
        self._retask()

    def _retask(self) -> None:
        """Hand the current task to a running policy.

        The same string conditions the policy and labels the recording, so a
        client that sets one does not have to remember to set the other.
        """
        if self._loaded is None:
            return
        if isinstance(self._control_worker, ModelControl):
            self._control_worker.task = self._task

    async def save_episode(self) -> None:
        """Close the open episode and write it.

        Blocking enough to be worth a thread: finalizing encodes video.

        Raises:
            RuntimeError: Nothing is recording.
        """
        mutation = self._recording.stop_episode()
        await asyncio.to_thread(mutation.save_episode)
        self._recording.mark_saved()

    async def discard_episode(self) -> None:
        """Drop the open episode's frames.

        Also how a client recovers from a failed save, which has already
        cleared the recording flag.

        Raises:
            RuntimeError: No dataset is open.
        """
        mutation = self._recording.current_mutation()
        if mutation is None:
            raise RuntimeError("No dataset is open")
        await asyncio.to_thread(mutation.discard_buffer)
        self._recording.mark_discarded()

    async def select_control(self, config: ControlConfig | None) -> None:
        """Choose what drives the followers, replacing whatever did.

        ``None`` leaves nothing driving, which is how a session starts. A choice
        made with no environment loaded is remembered and applied on load, like
        a dataset or a model.

        Raises:
            RuntimeError: Teleoperation with no leader, or a model with none loaded.
        """
        self._control = config
        if self._loaded is not None:
            await self._open_control()

    async def _open_control(self) -> None:
        """Run the chosen control against the loaded environment.

        The previous one is released first, so two are never writing the same
        action features. One that fails to start leaves none running.
        """
        await self._release_control()
        loaded = self._loaded
        if loaded is None or self._control is None:
            return
        control = self._build_control(self._control, loaded)
        stack = AsyncExitStack()
        try:
            await stack.enter_async_context(control)
        except BaseException:
            await stack.aclose()
            raise
        self._control_worker, self._control_stack = control, stack
        logger.info("{} now drives {}", control.name, loaded.name)

    async def _release_control(self) -> None:
        """Stop the running control, if any. Idempotent.

        Whatever it last wrote stays in the store, so a follower that is
        driving holds its last command rather than being left mid-ramp.
        """
        worker, self._control_worker = self._control_worker, None
        stack, self._control_stack = self._control_stack, None
        if stack is not None:
            await stack.aclose()
            logger.info("Released {} control", worker.name if worker else "?")

    def _build_control(self, config: ControlConfig, loaded: LoadedEnvironment) -> ControlAlgorithm:
        """The control ``config`` describes, wired to this environment's store.

        Raises:
            RuntimeError: Teleoperation without a leader, or a model with no
                policy loaded.
        """
        match config:
            case TeleopControlConfig():
                pair = loaded.pair_for_teleop()
                if pair is None:
                    raise RuntimeError(f"{loaded.name} has no leader to teleoperate from")
                leader, follower = pair
                return TeleopControl(leader, loaded.store, loaded.action_keys_for(follower), hz=config.hz)
            case ModelControlConfig():
                if self._policy is None:
                    raise RuntimeError("No model is loaded to drive with")
                return ModelControl(self._policy, loaded.store, loaded.shape, hz=config.hz, task=self._task)

    async def load_policy(
        self,
        model_id: UUID,
        path: Path,
        *,
        device: str,
    ) -> None:
        """Load a policy, replacing any already loaded.

        Checked against the shape of whatever environment is loaded, and opened
        lazily when none is -- the cameras it names come from the devices.

        Loading does not drive anything: a policy only reaches the arms once it
        is selected as the control.

        Raises:
            FileNotFoundError: No export at ``path``.
            ModelCameraMismatchError: It names a camera this session lacks.
        """
        await self.unload_policy()
        self._pending_model = (model_id, path, device)
        if self._loaded is not None:
            await self._open_policy()
        else:
            logger.info("Model {} selected; it will load when an environment is loaded", model_id)

    async def unload_policy(self) -> None:
        """Release the policy. Idempotent.

        Leaves nothing controlling, rather than guessing at teleoperation: the
        follower holds its last position until a client chooses what drives it.
        """
        self._pending_model = None
        self._model = self._policy = None
        if isinstance(self._control, ModelControlConfig):
            await self.select_control(None)

    async def _open_policy(self) -> None:
        """Load the pending policy against the loaded environment and select it.

        Selecting it displaces whatever was controlling the followers, which
        for a leader-and-follower environment means teleoperation stops. That
        is the point of there being one control: a human and a policy fighting
        over the same joints is not a state worth being able to reach.
        """
        if self._pending_model is None or self._loaded is None:
            return
        model_id, path, device = self._pending_model
        loaded, model = await asyncio.to_thread(load_model, model_id, path, self._loaded.shape, device=device)
        self._model, self._policy = loaded, model
        # Loaded, not selected. Selecting a control is what starts motion now
        # that there is no second switch, so loading a policy must not be the
        # thing that sets an arm going.
        if isinstance(self._control, ModelControlConfig):
            await self._open_control()

    async def unload(self) -> None:
        """Release the loaded environment. Idempotent.

        The control goes first: it writes action features the robots read, so
        stopping it before them means nothing is still being commanded while
        the arms are let go.
        """
        await self._release_control()
        stack, self._stack = self._stack, None
        loaded, self._loaded = self._loaded, None
        if stack is None:
            return
        # Cleared before closing, so a client asking mid-unload is told nothing
        # is loaded rather than handed devices on their way out.
        logger.info("Unloading {}", loaded.name if loaded else "environment")
        await stack.aclose()

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[RuntimeSession]:
        """Hold the session open, unloading anything still loaded on the way out."""
        try:
            yield self
        finally:
            await self.unload()
            await self.unload_dataset()
            await self.unload_policy()

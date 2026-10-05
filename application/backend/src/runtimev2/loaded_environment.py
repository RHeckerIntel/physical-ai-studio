"""One environment, loaded: its workers, and the store they talk through.

Separate from the session on purpose. A session is a client being connected; a
loaded environment is a set of devices held open with a store shaped to match
them. Tying the two together would make swapping environments mean dropping the
client, and the feature spec -- which the store is keyed by and which decides
what datasets and models fit -- changes with the environment, so the store has
to be rebuilt when it does.

The environment holds no devices and runs no threads itself. Each worker owns
its own device, its own thread and its own rate, so loading is building the
store and then entering workers, and unloading is leaving them.
"""

from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from runtimev2.control.teleop import check_pairing
from runtimev2.features import image_feature_key
from runtimev2.store import FeatureStore
from runtimev2.workers.base import ThreadedWorker
from runtimev2.workers.camera import CameraWorker
from runtimev2.workers.robot import RobotWorker
from workers.base import ManagedLifecycle

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Mapping

    from robots.robot_client_factory import RobotClientFactory
    from runtimev2.environment import CameraShape, RobotShape, SessionShape
    from runtimev2.leader import LeaderDevice
    from schemas.environment import EnvironmentWithRelations
    from schemas.project_camera import Camera as CameraRow
    from schemas.robot import ReadableRobot

# A follower is commanded as often as it can be, because that is what makes a
# slow control's ramp smooth rather than stepped.
DEFAULT_ROBOT_HZ = 100.0


@dataclass(frozen=True, slots=True)
class EnvironmentState:
    """What a loaded environment looks like from outside, for a client to render."""

    environment: str
    robots: dict[str, str] = field(default_factory=dict)
    """Robot key to role."""
    cameras: tuple[str, ...] = ()
    leaders: tuple[str, ...] = ()
    """Input devices. Separate from ``robots`` because nothing commands them."""
    features: int = 0


class LoadedEnvironment(ManagedLifecycle["LoadedEnvironment"]):
    """An environment held open: devices connected, store live, workers ticking.

    Opened with ``async with``, or through ``RuntimeSession.load``.
    """

    def __init__(
        self,
        environment: EnvironmentWithRelations,
        factory: RobotClientFactory,
        shape: SessionShape,
        leaders: Mapping[str, LeaderDevice],
        *,
        robot_hz: float = DEFAULT_ROBOT_HZ,
    ) -> None:
        """``shape`` and ``leaders`` are the session's, borrowed for this load.

        Neither is owned here: the shape is what datasets and models are
        checked against, and a leader outlasts any one environment.
        """
        self._environment = environment
        self._factory = factory
        self._robot_hz = robot_hz
        self._shape: SessionShape | None = shape
        self._leaders = leaders
        self._store: FeatureStore | None = None
        self._robots: dict[str, RobotWorker] = {}
        self._cameras: dict[str, CameraWorker] = {}

        # Workers that come and go while the devices stay connected -- a
        # dataset or a model is chosen above the environment and can be swapped
        # without dropping the arms. Each keeps its own teardown so it can be
        # detached on its own.
        self._attached: dict[str, tuple[ThreadedWorker, AsyncExitStack]] = {}

    @property
    def name(self) -> str:
        return self._environment.name

    @property
    def store(self) -> FeatureStore:
        """The environment's current truth.

        Raises:
            RuntimeError: The environment is not loaded.
        """
        if self._store is None:
            raise RuntimeError("Environment is not loaded")
        return self._store

    @property
    def shape(self) -> SessionShape:
        """The devices and features this environment contributes.

        Raises:
            RuntimeError: The environment is not loaded.
        """
        if self._shape is None:
            raise RuntimeError("Environment is not loaded")
        return self._shape

    @property
    def has_leader(self) -> bool:
        """Whether a leader arm is present to teleoperate from."""
        return bool(self._leaders)

    def state(self) -> EnvironmentState:
        """Summarize this environment for a client."""
        return EnvironmentState(
            environment=self._environment.name,
            robots={robot.key: robot.role for robot in self.shape.robots},
            cameras=tuple(self._cameras),
            leaders=tuple(self._leaders),
            features=len(self.store.spec.features),
        )

    @asynccontextmanager
    async def lifecycle(self) -> AsyncGenerator[LoadedEnvironment]:
        """Build the store, then enter every worker.

        Each worker acquires its own device and starts its own thread, so the
        only ordering this has to get right is that the store exists first.
        Leaving unwinds the workers in reverse, and each one stops ticking
        before releasing what it holds.
        """
        async with AsyncExitStack() as stack:
            self._store = FeatureStore(self.shape.feature_spec())
            logger.info(
                "Loading {} with {} robots, {} leaders, {} cameras and {} features",
                self._environment.name,
                len(self.shape.robots),
                len(self.shape.leaders),
                len(self.shape.cameras),
                len(self._store.spec.features),
            )
            try:
                for shape in self.shape.robots:
                    robot_worker = await self._robot_worker(shape)
                    await stack.enter_async_context(robot_worker)
                    self._robots[shape.key] = robot_worker
                for camera in self.shape.cameras:
                    camera_worker = self._camera_worker(camera)
                    await stack.enter_async_context(camera_worker)
                    self._cameras[camera.key] = camera_worker
                # Nothing controls a freshly loaded environment: the arms
                # publish what they measure and hold position until a client
                # says what should drive them, so loading never starts motion.
                # The pairing is still checked now rather than when teleop is
                # first selected, because an unpairable leader is a broken
                # environment and should say so on load.
                self.pair_for_teleop()
                yield self
            finally:
                # Detached first: an attached worker was entered on its own
                # stack, so unwinding this one would not reach it.
                for key in list(self._attached):
                    await self.detach(key)
                # Cleared before the stack unwinds, so a client reading state
                # mid-unload is not handed workers on their way out.
                self._robots.clear()
                self._cameras.clear()
        logger.info("Unloaded {}", self._environment.name)

    async def attach(self, key: str, worker: ThreadedWorker) -> None:
        """Run ``worker`` alongside the devices, replacing anything under ``key``.

        The previous one is detached first, so two workers cannot both be
        writing the same features. A worker that fails to start leaves nothing
        attached rather than a half-entered one.

        Raises:
            BaseException: Whatever the worker raised while acquiring.
        """
        await self.detach(key)
        stack = AsyncExitStack()
        try:
            await stack.enter_async_context(worker)
        except BaseException:
            await stack.aclose()
            raise
        self._attached[key] = (worker, stack)
        logger.info("Attached {} to {}", key, self._environment.name)

    async def detach(self, key: str) -> None:
        """Stop and release the worker under ``key``. Idempotent."""
        entry = self._attached.pop(key, None)
        if entry is None:
            return
        await entry[1].aclose()
        logger.info("Detached {} from {}", key, self._environment.name)

    def attached(self, key: str) -> ThreadedWorker | None:
        return entry[0] if (entry := self._attached.get(key)) else None

    async def _robot_worker(self, shape: RobotShape) -> RobotWorker:
        """Build an unconnected robot worker for ``shape``.

        A plain driver rather than a ``SharedRobot``: this worker is the only
        thing touching the arm while loaded, so an owner process would buy
        nothing and cost a zenoh hop on the 100Hz path. The session holds the
        port outright, which is what we want -- an arm being driven is not
        something a second caller should join.
        """
        robot, _definition = await self._factory.build_robot_driver(self._row_for(shape), self._factory)
        return RobotWorker(robot, self.store, shape=shape, hz=self._robot_hz)

    def _camera_worker(self, shape: CameraShape) -> CameraWorker:
        """Build a camera worker for ``shape``.

        It takes a builder rather than a camera because attaching is retried.
        ``validate_on_connect`` stays off: the environment's declared resolution
        leads and the worker resizes to it, so an IP camera serving its own size
        is usable. ``overwrite_settings`` stays off too, so loading an
        environment cannot reconfigure a camera another session is watching.
        """
        from utils.camera_factory import build_shared_camera

        row = self._camera_row_for(shape)
        return CameraWorker(
            lambda: build_shared_camera(row, validate_on_connect=False, overwrite_settings=False),
            self.store,
            key=image_feature_key(shape.key),
            hz=shape.fps,
        )

    def _row_for(self, shape: RobotShape) -> ReadableRobot:
        """Find the database row a described robot came from.

        Raises:
            RuntimeError: The shape names a robot the environment does not hold.
        """
        for configured in self._environment.robots:
            for candidate in (configured.robot, getattr(configured.tele_operator, "robot", None)):
                if candidate is not None and str(candidate.id) == shape.robot_id:
                    return candidate
        raise RuntimeError(f"Robot {shape.key} is not part of {self._environment.name}")

    def _camera_row_for(self, shape: CameraShape) -> CameraRow:
        """Find the database row a described camera came from.

        Raises:
            RuntimeError: The shape names a camera the environment does not hold.
        """
        for candidate in self._environment.cameras:
            if str(candidate.id) == shape.camera_id:
                return candidate
        raise RuntimeError(f"Camera {shape.key} is not part of {self._environment.name}")

    def pair_for_teleop(self) -> tuple[LeaderDevice, RobotShape] | None:
        """The leader and follower teleoperation would use, if there are any.

        ``None`` when there is no leader, which is a normal environment.

        Raises:
            RuntimeError: Several of either -- the environment does not say
                which drives which -- or joint counts that cannot correspond.
        """
        followers = self.shape.robots
        if not self._leaders or not followers:
            return None
        if len(self._leaders) != 1 or len(followers) != 1:
            raise RuntimeError(
                f"cannot pair {len(self._leaders)} leaders with {len(followers)} followers; "
                "the environment does not say which drives which"
            )
        leader = next(iter(self._leaders.values()))
        follower = followers[0]
        check_pairing(leader.joint_names, follower.joint_names)
        return leader, follower

    def action_keys_for(self, shape: RobotShape) -> tuple[str, ...]:
        """The action features a control would write to drive ``shape``."""
        return self._robots[shape.key].action_keys

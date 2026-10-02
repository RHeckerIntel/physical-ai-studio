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

from runtimev2.environment import describe_environment
from runtimev2.features import ACTION_PREFIX, OBSERVATION_PREFIX, image_feature_key, joint_feature_key
from runtimev2.store import FeatureStore
from runtimev2.workers.camera import CameraWorker
from runtimev2.workers.robot import RobotWorker
from runtimev2.workers.teleop import TeleopSource, joint_mapping
from workers.base import ManagedLifecycle

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from robots.robot_client_factory import RobotClientFactory
    from runtimev2.environment import CameraShape, EnvironmentShape, RobotShape
    from schemas.environment import EnvironmentWithRelations
    from schemas.project_camera import Camera as CameraRow
    from schemas.robot import ReadableRobot

# A leader is read as fast as it will answer, which is what makes teleoperation
# feel direct. A follower is written at the same rate it is read.
DEFAULT_ROBOT_HZ = 100.0


@dataclass(frozen=True, slots=True)
class EnvironmentState:
    """What a loaded environment looks like from outside, for a client to render."""

    environment: str
    robots: dict[str, str] = field(default_factory=dict)
    """Robot key to role."""
    cameras: tuple[str, ...] = ()
    teleoperating: bool = False
    features: int = 0


class LoadedEnvironment(ManagedLifecycle["LoadedEnvironment"]):
    """An environment held open: devices connected, store live, workers ticking.

    Opened with ``async with``, or through ``RuntimeSession.load``.
    """

    def __init__(
        self,
        environment: EnvironmentWithRelations,
        factory: RobotClientFactory,
        *,
        robot_hz: float = DEFAULT_ROBOT_HZ,
    ) -> None:
        self._environment = environment
        self._factory = factory
        self._robot_hz = robot_hz
        self._shape: EnvironmentShape | None = None
        self._store: FeatureStore | None = None
        self._robots: dict[str, RobotWorker] = {}
        self._cameras: dict[str, CameraWorker] = {}
        self._teleop: TeleopSource | None = None

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
    def shape(self) -> EnvironmentShape:
        """The devices and features this environment contributes.

        Raises:
            RuntimeError: The environment is not loaded.
        """
        if self._shape is None:
            raise RuntimeError("Environment is not loaded")
        return self._shape

    @property
    def has_leader(self) -> bool:
        return self._teleop is not None

    def state(self) -> EnvironmentState:
        """Summarize this environment for a client."""
        return EnvironmentState(
            environment=self._environment.name,
            robots={robot.key: robot.role for robot in self.shape.robots},
            cameras=tuple(self._cameras),
            teleoperating=any(worker.write_actions for worker in self._robots.values()),
            features=len(self.store.spec.features),
        )

    def set_teleoperating(self, enabled: bool) -> None:
        """Let the followers follow, or stop them.

        Only the write side is switched. The leader keeps being read and the
        mapping keeps publishing either way, so a client can watch what would
        be commanded before committing to it -- and enabling is then one flag
        rather than a sequence that has to be started in the right order.

        Raises:
            RuntimeError: This environment has no leader.
        """
        if self._teleop is None:
            raise RuntimeError("This environment has no leader to teleoperate from")
        for robot in self.shape.followers:
            self._robots[robot.key].write_actions = enabled
        logger.info("Teleoperation {} for {}", "enabled" if enabled else "disabled", self._environment.name)

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[LoadedEnvironment]:
        """Build the store, then enter every worker.

        Each worker acquires its own device and starts its own thread, so the
        only ordering this has to get right is that the store exists first.
        Leaving unwinds the workers in reverse, and each one stops ticking
        before releasing what it holds.
        """
        async with AsyncExitStack() as stack:
            self._shape = await describe_environment(self._environment, self._factory)
            self._store = FeatureStore(self._shape.feature_spec())
            logger.info(
                "Loading {} with {} robots, {} cameras and {} features",
                self._environment.name,
                len(self._shape.robots),
                len(self._shape.cameras),
                len(self._store.spec.features),
            )
            try:
                for shape in self._shape.robots:
                    robot_worker = await self._robot_worker(shape)
                    await stack.enter_async_context(robot_worker)
                    self._robots[shape.key] = robot_worker
                for camera in self._shape.cameras:
                    camera_worker = self._camera_worker(camera)
                    await stack.enter_async_context(camera_worker)
                    self._cameras[camera.key] = camera_worker
                teleop = self._build_teleop()
                if teleop is not None:
                    await stack.enter_async_context(teleop)
                    self._teleop = teleop
                yield self
            finally:
                # Cleared before the stack unwinds, so a client reading state
                # mid-unload is not handed workers on their way out.
                self._robots.clear()
                self._cameras.clear()
                self._teleop = None
        logger.info("Unloaded {}", self._environment.name)

    async def _robot_worker(self, shape: RobotShape) -> RobotWorker:
        """Build an unconnected robot worker for ``shape``.

        The robot itself is built here because that is async and needs the
        factory; connecting it is the worker's own business.
        """
        robot, _definition = await self._factory.build_shared_robot(self._row_for(shape))
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

    def _build_teleop(self) -> TeleopSource | None:
        """Map each leader's observations onto a follower's actions.

        Returns ``None`` when the environment has no leader, which is a normal
        environment rather than a problem -- one with no control source still
        publishes observations.

        Raises:
            RuntimeError: Several leaders or followers, which the environment
                does not say how to pair.
        """
        leaders = [robot for robot in self.shape.robots if robot.role == "leader"]
        followers = self.shape.followers
        if not leaders or not followers:
            return None
        if len(leaders) != 1 or len(followers) != 1:
            raise RuntimeError(
                f"cannot pair {len(leaders)} leaders with {len(followers)} followers; "
                "the environment does not say which drives which"
            )
        leader, follower = leaders[0], followers[0]
        mapping = joint_mapping(
            tuple(joint_feature_key(OBSERVATION_PREFIX, leader.key, joint) for joint in leader.joint_names),
            tuple(joint_feature_key(ACTION_PREFIX, follower.key, joint) for joint in follower.joint_names),
        )
        # At the robots' rate: a command is only as fresh as the leader it came from.
        return TeleopSource(self.store, mapping, hz=self._robot_hz)

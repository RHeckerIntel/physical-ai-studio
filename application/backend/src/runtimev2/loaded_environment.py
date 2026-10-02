"""One environment, loaded: its devices, its store and the threads ticking them.

Separate from the session on purpose. A session is a client being connected; a
loaded environment is a set of robots held open with a store shaped to match
them. Tying the two together would make swapping environments mean dropping the
client, and the feature spec -- which the store is keyed by and which decides
what datasets and models fit -- changes with the environment, so the store has
to be rebuilt when it does.

Everything here is released on the way out, in reverse: the threads stop before
the robots they drive are disconnected.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from runtimev2.environment import describe_environment
from runtimev2.features import ACTION_PREFIX, OBSERVATION_PREFIX, joint_feature_key
from runtimev2.store import FeatureStore
from runtimev2.workers.loop import run_at
from runtimev2.workers.robot import RobotWorker
from runtimev2.workers.teleop import TeleopSource, joint_mapping
from workers.base import ManagedLifecycle

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from physicalai.robot.interface import Robot

    from robots.robot_client_factory import RobotClientFactory
    from runtimev2.environment import EnvironmentShape, RobotShape
    from schemas.environment import EnvironmentWithRelations
    from schemas.robot import ReadableRobot

# A leader is read as fast as it will answer, which is what makes teleoperation
# feel direct. A follower is written at the same rate it is read.
DEFAULT_ROBOT_HZ = 100.0
# Long enough for a tick in flight to finish, short enough that unloading does
# not feel stuck.
_THREAD_JOIN_TIMEOUT_S = 5.0


@dataclass(frozen=True, slots=True)
class EnvironmentState:
    """What a loaded environment looks like from outside, for a client to render."""

    environment: str
    robots: dict[str, str] = field(default_factory=dict)
    """Robot key to role."""
    teleoperating: bool = False
    features: int = 0


class LoadedEnvironment(ManagedLifecycle["LoadedEnvironment"]):
    """An environment held open: robots connected, store live, workers ticking.

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
        self._workers: dict[str, RobotWorker] = {}
        self._teleop: TeleopSource | None = None
        self._stop = threading.Event()

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
            teleoperating=any(worker.write_actions for worker in self._workers.values()),
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
            self._workers[robot.key].write_actions = enabled
        logger.info("Teleoperation {} for {}", "enabled" if enabled else "disabled", self._environment.name)

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[LoadedEnvironment]:
        """Connect the devices, start the workers, and give all of it back after.

        Each piece is pushed onto the stack as it is acquired, so a failure part
        way through releases exactly what was taken.
        """
        async with AsyncExitStack() as stack:
            self._shape = await describe_environment(self._environment, self._factory)
            self._store = FeatureStore(self._shape.feature_spec())
            logger.info(
                "Loading {} with {} robots and {} features",
                self._environment.name,
                len(self._shape.robots),
                len(self._store.spec.features),
            )

            robots = {
                robot.key: await stack.enter_async_context(self._connected(robot)) for robot in self._shape.robots
            }
            self._workers = {key: RobotWorker(robot, self._store, name=key) for key, robot in robots.items()}
            self._teleop = self._build_teleop()
            stack.enter_context(self._running())
            yield self
        logger.info("Unloaded {}", self._environment.name)

    @asynccontextmanager
    async def _connected(self, shape: RobotShape) -> AsyncIterator[Robot]:
        """Connect one robot for the body of the block, and disconnect it after.

        Both calls go off the event loop: connecting spawns the owner process
        that holds the hardware, and disconnecting waits for it to let go.
        """
        row = self._row_for(shape)
        robot, _definition = await self._factory.build_shared_robot(row)
        await asyncio.to_thread(robot.connect)
        try:
            self._verify_joint_order(shape, robot)
            yield robot
        finally:
            await asyncio.to_thread(robot.disconnect)

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

    @staticmethod
    def _verify_joint_order(shape: RobotShape, robot: Robot) -> None:
        """Fail if the connected robot disagrees with the shape it was described by.

        The spec's joint order comes from a locally constructed driver; the
        device's comes from the owner process's metadata. An action vector is
        assembled in the spec's order, so a disagreement would send joint
        values to the wrong joints -- silently, and while the arm is live.

        Raises:
            RuntimeError: The orders differ.
        """
        connected = tuple(robot.joint_names)
        if connected != shape.joint_names:
            raise RuntimeError(
                f"Robot {shape.key} reports joints {connected} once connected, but was described as {shape.joint_names}"
            )

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
        return TeleopSource(self.store, mapping)

    @contextmanager
    def _running(self) -> Iterator[None]:
        """Tick every worker in its own thread for the body of the block.

        Daemon threads, but they are still joined: a thread holding a robot
        mid-write has to finish before the robot is disconnected under it.
        """
        self._stop.clear()
        tickers: list[Any] = [*self._workers.values()]
        if self._teleop is not None:
            tickers.append(self._teleop)
        threads = [
            threading.Thread(
                target=run_at,
                args=(ticker, self._robot_hz, self._stop.is_set),
                name=f"runtimev2-{ticker.name}",
                daemon=True,
            )
            for ticker in tickers
        ]
        for thread in threads:
            thread.start()
        try:
            yield
        finally:
            self._stop.set()
            for thread in threads:
                thread.join(_THREAD_JOIN_TIMEOUT_S)
                if thread.is_alive():
                    logger.warning("{} did not stop within {}s", thread.name, _THREAD_JOIN_TIMEOUT_S)

#!/usr/bin/env python3

from workers.base import StoppableMixin
from dataclasses import dataclass
from abc import abstractmethod
from typing import AsyncGenerator
from contextlib import asynccontextmanager, AbstractAsyncContextManager
from schemas.environment import TeleoperatorRobotWithRobot
from physicalai.robot.transport._owner_config import normalize_robot_config
from physicalai.config import Config
from utils.serial_robot_tools import RobotConnectionManager
from robots.robot_client_factory import RobotClientFactory
import multiprocessing as mp
from multiprocessing.synchronize import Event as EventClass
from workers.base import run_at_frequency
from physicalai.robot.interface import Robot  # noqa: PLC0415
from typing import Generator
from contextlib import contextmanager
import asyncio

from db.engine import async_session
from services.environment_service import EnvironmentService, RobotCatalogRegistry
from uuid import UUID

from loguru import logger


class BaseProcessWorker[Context](mp.Process, StoppableMixin):
    def __init__(self,  stop_event: EventClass):
        super().__init__()
        self._interrupt_event = stop_event
        self._stop_event = mp.Event()

    @abstractmethod
    def lifecycle(self) -> AbstractAsyncContextManager[Context]:
        """Acquire everything, yield to run loop and then clean up."""

    @abstractmethod
    async def run_loop(self, context: Context) -> None:
        """Run loop of process."""

    def run(self) -> None:
        with logger.contextualize(worker=self.__class__.__name__):
            try:
                logger.info(f"Starting {self.name}")
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                async def _main() -> None:
                    async with self.lifecycle() as ctx:
                        await self.run_loop(ctx)
                self.loop.run_until_complete(_main())
                logger.info(f"Stopped {self.name}")
            except Exception:
                logger.exception(f"Unhandled exception in {self.name}")
            finally:
                self.loop.run_until_complete(self.loop.shutdown_asyncgens())
                self.loop.close()
                logger.info(f"Stopped {self.name}.")

    def stop(self) -> None:
        timeout = 10
        self._stop_event.set()
        if not self.is_alive():
            return
        self.join(timeout=timeout)
        if self.is_alive():
            logger.warning(f"Process {self.name} did not stop within {timeout}s, terminating")
            self.terminate()
            self.join(timeout=2.0)

@dataclass(frozen=True, slots=True)
class TeleopContext:
    follower: Robot | None
    leader: Robot| None

class TeleoperateWorker(BaseProcessWorker[TeleopContext]):
    def __init__(
        self,
        follower_config: Config,
        leader_config: Config,
        stop_event: EventClass
    ):
        super().__init__(stop_event=stop_event)
        self.follower_config = follower_config
        self.leader_config = leader_config
        self.follower: Robot | None = None
        self.leader: Robot | None = None

    @staticmethod
    def _load_robot_from_config(config: Config) -> Robot | None:
        robot = normalize_robot_config(config).instantiate(expected_type=Robot)
        if isinstance(robot, Robot):
            return robot

    @asynccontextmanager
    async def lifecycle(self) -> AsyncGenerator[TeleopContext]:
        follower = self._load_robot_from_config(self.follower_config)
        if follower:
            follower.connect()

        leader = self._load_robot_from_config(self.leader_config)
        if leader:
            leader.connect()

        yield TeleopContext(
            follower=follower,
            leader=leader
        )
        if follower:
            follower.disconnect()

        if leader:
            leader.disconnect()


    async def run_loop(self, context: TeleopContext):
        i = 0
        if context.follower is None:
            return

        while not self.should_stop():
            async with run_at_frequency(100):

                if context.leader:
                    obs = context.leader.get_observation()
                    context.follower.send_action(obs.joint_positions)
                i += 1
                if i > 1000:
                    break #raise Exception("End of the test run... throwing error")


@contextmanager
def run_managed_process(worker: BaseProcessWorker) -> Generator[None]:
    try:
        worker.start()
        yield
    finally:
        worker.stop()


@dataclass(frozen=True, slots=True)
class RuntimeSession:
    name: str
    interrupt_event: EventClass


async def main():
    session  =async_session()
    project_id = UUID("59498fc1-cdba-4c7e-856b-d5653d3b6640")
    environment_id = UUID("1d20ee3a-4e58-4917-8333-6a8c3b79565a")
    catalog_registry = RobotCatalogRegistry()
    environment_service = EnvironmentService(session=session, catalog_registry=catalog_registry)
    robot_factory= RobotClientFactory(robot_manager = RobotConnectionManager(), catalog_registry=catalog_registry)
    env = await environment_service.get_environment_by_id(project_id, environment_id)
    robot = env.robots[0]
    follower_driver, _ = await robot_factory.build_robot_driver(robot.robot, robot_factory)
    follower_config = Config.from_instance(follower_driver)

    if isinstance(robot.tele_operator, TeleoperatorRobotWithRobot):
        leader_driver, _ = await robot_factory.build_robot_driver(robot.tele_operator.robot, robot_factory)
        leader_config = Config.from_instance(leader_driver)

        print("Start")
        worker = TeleoperateWorker(follower_config, leader_config, stop_event=mp.Event())
        with run_managed_process(worker):
            i = 0
            while worker.is_alive():
                i += 1
                await asyncio.sleep(0.01)
                #if i > 10:
                #    print("closing from main side...")
                #    break

        print("done")
    await session.close()

asyncio.run(main())

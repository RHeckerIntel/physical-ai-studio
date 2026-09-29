from workers.base import run_at_frequency
from schemas.robot import ReadableRobot
from physicalai.robot.transport._owner_config import normalize_robot_config
from physicalai.robot import Robot
from physicalai.config import Config
from typing import  AsyncGenerator, Dict, Any
import zenoh
from multiprocessing.synchronize import Event as EventClass

from dataclasses import dataclass
from contextlib import asynccontextmanager, AsyncExitStack
from runtimev2.thread_base import BaseThreadWorker

@dataclass(frozen=True, slots=True)
class EnvironmentContext:
    stack: AsyncExitStack
    robot: ReadableRobot
    leader: ReadableRobot | None
    #publishers: dict[str, zenoh.Publisher]
    listeners: list[zenoh.Subscriber]

class EnvironmentWorker(BaseThreadWorker[EnvironmentContext]):

    def __init__(self, document: Dict[str, Any], zenoh_key: zenoh.KeyExpr, stop_event: EventClass ):
        super().__init__(stop_event=stop_event)
        self.document = document
        self.zenoh_key = zenoh_key

    @asynccontextmanager
    async def lifecycle(self) -> AsyncGenerator[EnvironmentContext]:
        async with AsyncExitStack() as stack:
            zenoh_session = stack.enter_context(zenoh.open(zenoh.Config()))
            robot = self._load_robot_from_config(self.document["robot"])
            if robot is None:
                raise ValueError("Excepted robot to control")
            robot.connect()
            leader = self._load_robot_from_config(self.document["leader"])
            if leader:
                leader.connect()


            yield EnvironmentContext(
                robot = robot,
                leader = leader,
                listeners=[],
                stack=stack
            )
            robot.disconnect()
            if leader:
                leader.disconnect()

    async def run_loop(self, context: EnvironmentContext):
        async with run_at_frequency(100):
            while not self.should_stop():
                if context.leader:
                    obs = context.leader.get_observation()
                    context.robot.send_action(obs.joint_positions)


    @staticmethod
    def _load_robot_from_config(config: Config) -> Robot | None:
        robot = normalize_robot_config(config).instantiate(expected_type=Robot)
        if isinstance(robot, Robot):
            return robot


#
#def generate_environment(document: Config, zenoh_key: zenoh.KeyExpr) -> None:
#    document.

def _load_robot_from_config(config: Config) -> Robot | None:
    robot = normalize_robot_config(config).instantiate(expected_type=Robot)
    if isinstance(robot, Robot):
        return robot

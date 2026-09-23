from contextlib import AsyncExitStack
from runtimev2.thread_base import BaseThreadWorker
from uuid import uuid4
from typing import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from dataclasses import dataclass
from multiprocessing.synchronize import Event as EventClass
from runtimev2.zenoh_anchor import zenoh_anchor
import zenoh

@dataclass(frozen=True, slots=True)
class RuntimeSessionContext:
    stack: AsyncExitStack
    listeners: list[zenoh.Subscriber]



class RuntimeSession(BaseThreadWorker[RuntimeSessionContext]):
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.zenoh_key = str(uuid4())

    @asynccontextmanager
    async def lifecycle(self) -> AsyncGenerator[RuntimeSessionContext]:
        async with AsyncExitStack() as stack:
            zenoh_session = stack.enter_context(zenoh.open(zenoh.Config()))
            yield RuntimeSessionContext(
                stack=stack,
                listeners=[
                    zenoh_session.declare_subscriber(f"{self.zenoh_key}/load_environment", self._load_environment)
                ]
            )


    async def run_loop(self, context: RuntimeSessionContext):
        pass

    def _load_environment(self, sample: zenoh.Sample):
        print("sample")
        print(sample)

@asynccontextmanager
async def runtime_session(config: dict[str, Any]) -> AsyncGenerator[RuntimeSession]:
    session = RuntimeSession(config)
    yield session
    #robot = normalize_robot_config(config).instantiate(expected_type=Robot)
    #robot.connect()
    #try:
    #    yield robot
    #finally:
    #    robot.disconnect()

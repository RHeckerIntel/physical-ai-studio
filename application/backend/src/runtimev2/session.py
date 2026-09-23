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
    zenoh_pub: zenoh.Publisher
    zenoh_sub: zenoh.Subscriber
    stack: AsyncExitStack



class RuntimeSession(BaseThreadWorker[RuntimeSessionContext]):
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.zenoh_key = str(uuid4())

    @asynccontextmanager
    async def lifecycle(self) -> AsyncGenerator[RuntimeSessionContext]:
        async with AsyncExitStack() as stack:
            zenoh_pub, zenoh_sub = await stack.enter_async_context(zenoh_anchor(self.zenoh_key, self._on_message))
            yield RuntimeSessionContext(zenoh_pub=zenoh_pub, zenoh_sub=zenoh_sub, stack=stack)


    async def run_loop(self, context: RuntimeSessionContext):
        pass

    def _on_message(self, sample: zenoh.Sample):
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

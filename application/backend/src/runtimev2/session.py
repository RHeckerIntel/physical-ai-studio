from runtimev2.components.environment import EnvironmentWorker
import msgpack

from runtimev2.thread_base import BaseThreadWorker
from uuid import uuid4
from typing import AsyncGenerator
from contextlib import asynccontextmanager, contextmanager, AsyncExitStack
from typing import Any
from dataclasses import dataclass
from multiprocessing.synchronize import Event as EventClass
import asyncio
import zenoh

@dataclass(frozen=True, slots=True)
class RuntimeSessionContext:
    stack: AsyncExitStack
    #publishers: dict[str, zenoh.Publisher]
    listeners: list[zenoh.Subscriber]



class RuntimeSession(BaseThreadWorker[RuntimeSessionContext]):
    def __init__(self, config: dict[str, Any], stop_event: EventClass):
        super().__init__(stop_event=stop_event)
        self.config = config
        self.zenoh_key = zenoh.KeyExpr(f"{uuid4()}")
        self.environment : EnvironmentWorker | None = None

    @asynccontextmanager
    async def lifecycle(self) -> AsyncGenerator[RuntimeSessionContext]:
        async with AsyncExitStack() as stack:
            zenoh_session = stack.enter_context(zenoh.open(zenoh.Config()))
            self.environment : EnvironmentWorker | None = None
            print("setting up context...")
            yield RuntimeSessionContext(
                stack=stack,
                #publishers={
                #    "load_environment": zenoh_session.declare_publisher(self.zenoh_key.concat("load_environment"))
                #},
                listeners=[
                    zenoh_session.declare_subscriber(self.zenoh_key.concat("/load_environment"), self._load_environment)
                ]
            )
            print("End of runtimeSessionContext")
            if self.environment:
                print("Stopping environment...")
                self.environment.stop()

    def _load_environment(self, sample: zenoh.Sample):
        print("load environment gotten:")
        if self.environment:
            self.environment.stop()

        data = sample.payload.to_bytes()
        payload = msgpack.unpackb(data, raw=False)
        self.environment = EnvironmentWorker(payload, self.zenoh_key, stop_event=self._interrupt_event)
        self.environment.start()

    async def run_loop(self, context: RuntimeSessionContext):
        while not self.should_stop():
            print("run loop context...")
            await asyncio.sleep(1 / 30)

    @contextmanager
    def external_interface(self):
        with zenoh.open(zenoh.Config()) as zenoh_session:
            import time
            yield {
                "load_environment": zenoh_session.declare_publisher(self.zenoh_key.concat("/load_environment"))
            }


@asynccontextmanager
async def runtime_session(config: dict[str, Any], stop_event: EventClass) -> AsyncGenerator[RuntimeSession]:
    session = RuntimeSession(config, stop_event=stop_event)
    session.start()
    await session.wait_for_loaded()
    yield session
    session.stop()

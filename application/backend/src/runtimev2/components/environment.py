import zenoh
from multiprocessing.synchronize import Event as EventClass

from dataclasses import dataclass
from contextlib import asynccontextmanager, AsyncGenerator, AsyncExitStack
from runtimev2.thread_base import BaseThreadWorker
from contextlib import asynccontextmanager

@dataclass(frozen=True, slots=True)
class EnvironmentContext:
    stack: AsyncExitStack
    #publishers: dict[str, zenoh.Publisher]
    listeners: list[zenoh.Subscriber]

class EnvironmentWorker(BaseThreadWorker[EnvironmentContext]):

    def __init__(self, zenoh_key: zenoh.KeyExpr, stop_event: EventClass ):
        super().__init__(stop_event=stop_event)
        self.zenoh_key = zenoh_key

    @asynccontextmanager
    async def lifecycle(self) -> AsyncGenerator[EnvironmentContext]:
        async with AsyncExitStack() as stack:
            zenoh_session = stack.enter_context(zenoh.open(zenoh.Config()))
            yield EnvironmentContext(
                listeners=[],
                stack=stack
            )

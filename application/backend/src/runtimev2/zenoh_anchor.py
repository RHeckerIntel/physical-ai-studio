from typing import Tuple
from typing import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Callable
import zenoh


@asynccontextmanager
async def zenoh_anchor(key: str, listener: Callable[[zenoh.Sample],None]) -> AsyncGenerator[Tuple[zenoh.Publisher, zenoh.Subscriber]]:
    with zenoh.open(zenoh.Config()) as session:
        sub = session.declare_subscriber(key, listener)
        pub = session.declare_publisher(key)
        yield pub, sub

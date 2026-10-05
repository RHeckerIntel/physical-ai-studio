"""Process workers a spawned child can rebuild, for testing the base class.

At module scope so they pickle: a class defined inside a test function cannot
cross a spawn boundary.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from runtimev2.workers.process import ProcessWorker

if TYPE_CHECKING:
    from collections.abc import Generator
    from multiprocessing.synchronize import Event as EventType


class CountingWorker(ProcessWorker):
    """Sets an event once it has ticked, so the parent can tell it ran."""

    def __init__(self, ticked: EventType, *, hz: float = 200.0) -> None:
        super().__init__(name="counting", hz=hz)
        self._ticked = ticked

    @contextmanager
    def acquire(self) -> Generator[None]:
        yield

    def tick(self) -> None:
        self._ticked.set()


class HeldWorker(ProcessWorker):
    """Signals from inside ``acquire``, so holding and releasing are observable."""

    def __init__(self, held: EventType, released: EventType) -> None:
        super().__init__(name="held", hz=100.0)
        self._held = held
        self._released = released

    @contextmanager
    def acquire(self) -> Generator[None]:
        self._held.set()
        try:
            yield
        finally:
            self._released.set()

    def tick(self) -> None:
        return


class RefusingWorker(ProcessWorker):
    """Fails in ``acquire``, which the parent must be told about."""

    def __init__(self) -> None:
        super().__init__(name="refusing", hz=100.0)

    @contextmanager
    def acquire(self) -> Generator[None]:
        raise RuntimeError("this device refused to open")
        yield  # pragma: no cover

    def tick(self) -> None:
        return  # pragma: no cover


class DyingWorker(ProcessWorker):
    """Fails on its first tick, after having been held."""

    def __init__(self) -> None:
        super().__init__(name="dying", hz=200.0)

    @contextmanager
    def acquire(self) -> Generator[None]:
        yield

    def tick(self) -> None:
        raise RuntimeError("the device went away mid-run")

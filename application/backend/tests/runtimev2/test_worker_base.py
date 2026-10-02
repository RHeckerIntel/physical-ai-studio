"""A worker is a thread: entering it starts the thread and waits for its device."""

from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

from runtimev2.workers.base import ThreadedWorker

if TYPE_CHECKING:
    from collections.abc import Generator


class _Spy(ThreadedWorker):
    """Records the order it was taken through its lifecycle."""

    def __init__(self, *, hz: float = 200.0, acquire_error: Exception | None = None) -> None:
        super().__init__(name="spy", hz=hz)
        self.events: list[str] = []
        self.ticks = 0
        self.acquire_error = acquire_error
        self.tick_error: Exception | None = None
        self.acquired_on: str | None = None

    @contextmanager
    def acquire(self) -> Generator[None]:
        if self.acquire_error is not None:
            raise self.acquire_error
        self.acquired_on = threading.current_thread().name
        self.events.append("acquired")
        try:
            yield
        finally:
            self.events.append("released")

    def tick(self) -> None:
        self.ticks += 1
        if self.tick_error is not None:
            raise self.tick_error


class TestLifecycle:
    async def test_entering_holds_the_device_and_ticking_starts(self) -> None:
        worker = _Spy()

        async with worker:
            assert worker.events == ["acquired"], "yielded before the device was held"
            await asyncio.sleep(0.05)
            assert worker.ticks > 0

        assert worker.events == ["acquired", "released"]
        assert not worker.is_alive()

    async def test_the_device_is_acquired_on_the_workers_own_thread(self) -> None:
        """That is the point of inheriting from Thread: connecting blocks here,
        not on the caller's event loop."""
        worker = _Spy()

        async with worker:
            pass

        assert worker.acquired_on == "spy"

    async def test_ticking_stops_before_the_device_is_released(self) -> None:
        """A tick must never run against a device that has been let go."""
        worker = _Spy()

        async with worker:
            await asyncio.sleep(0.05)
        ticks_at_exit = worker.ticks

        await asyncio.sleep(0.05)
        assert worker.ticks == ticks_at_exit

    async def test_it_is_a_thread(self) -> None:
        worker = _Spy()

        assert isinstance(worker, threading.Thread)
        assert worker.daemon, "a wedged worker must not keep the interpreter alive"


class TestFailures:
    async def test_a_failure_to_acquire_surfaces_to_whoever_entered(self) -> None:
        """Otherwise it would be lost on a thread nobody is watching, and the
        load would look like it succeeded."""
        worker = _Spy(acquire_error=RuntimeError("device or resource busy"))

        with pytest.raises(RuntimeError, match="busy"):
            async with worker:
                pytest.fail("the body must not run")

        assert not worker.is_alive()

    async def test_a_failing_tick_stops_the_worker_and_is_reported(self) -> None:
        worker = _Spy()
        worker.tick_error = RuntimeError("publisher went away")

        async with worker:
            await asyncio.sleep(0.05)
            assert not worker.is_alive(), "the thread kept going after a failed tick"

        assert worker.events == ["acquired", "released"], "the device was not released"

    async def test_a_failure_to_acquire_does_not_hang_the_caller(self) -> None:
        """The ready event has to be set even when acquiring raised."""
        worker = _Spy(acquire_error=RuntimeError("nope"))

        with pytest.raises(RuntimeError):
            await asyncio.wait_for(worker.__aenter__(), timeout=5)


class TestSingleUse:
    async def test_a_worker_cannot_be_entered_twice(self) -> None:
        """Like the thread it is. Loading an environment builds fresh workers."""
        worker = _Spy()

        async with worker:
            pass

        with pytest.raises(RuntimeError):
            async with worker:
                pass

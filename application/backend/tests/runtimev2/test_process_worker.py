"""ProcessWorker is a ThreadedWorker that happens to be a process.

Same ``acquire``/``tick``/``async with``, so a device loop can be moved between
the two without being rewritten.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp

import pytest

from tests.runtimev2.fake_process_worker import CountingWorker, DyingWorker, HeldWorker, RefusingWorker
from utils.multiprocessing import ensure_spawn_start_method


@pytest.fixture(autouse=True)
def _spawn() -> None:
    ensure_spawn_start_method()


class TestTheSameShapeAsAThread:
    async def test_entering_starts_it_and_leaving_ends_it(self) -> None:
        ticked = mp.Event()
        worker = CountingWorker(ticked)

        async with worker:
            assert worker.is_alive()
            await asyncio.to_thread(ticked.wait, 10)

        assert ticked.is_set(), "the child never ticked"
        assert not worker.is_alive()

    async def test_it_yields_only_once_the_device_is_held(self) -> None:
        """A caller past ``async with`` knows the worker is live."""
        held, released = mp.Event(), mp.Event()
        worker = HeldWorker(held, released)

        async with worker:
            assert held.is_set()
            assert not released.is_set()

        await asyncio.to_thread(released.wait, 10)
        assert released.is_set(), "acquire's teardown never ran"

    def test_the_rate_is_whatever_it_was_given(self) -> None:
        assert CountingWorker(mp.Event(), hz=12.0).hz == 12.0


class TestFailures:
    async def test_a_refused_device_fails_the_caller(self) -> None:
        """Where it can fail a load, rather than in a child nobody watches."""
        with pytest.raises(RuntimeError, match="refused to open"):
            async with RefusingWorker():
                pass

    async def test_the_reason_crosses_the_boundary(self) -> None:
        """An exception object cannot, so the text does."""
        with pytest.raises(RuntimeError, match="failed to start"):
            async with RefusingWorker():
                pass

    async def test_dying_mid_run_is_reported_not_raised(self) -> None:
        """The caller is already past ``async with``; it is logged instead."""
        worker = DyingWorker()

        async with worker:
            await asyncio.to_thread(worker.join, 10)

        assert not worker.is_alive()

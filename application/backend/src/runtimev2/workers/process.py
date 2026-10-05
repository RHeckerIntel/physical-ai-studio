"""A worker that owns its device, its *process* and its rate.

:class:`~runtimev2.workers.base.ThreadedWorker`'s shape with a process behind
it: same ``acquire``, ``tick`` and ``async with``, so a device loop moves
between the two without being rewritten. Worth moving one when it must not be
descheduled -- a robot missing ten deadlines per recording cycle in-process
missed none in its own, with no priority granted.

Two differences, both forced by the boundary. *The worker is pickled*, so a
subclass holds only what pickles and builds its device in ``acquire``, which
runs in the child. *A failure travels as text*, since an exception object
cannot; the traceback stays in the child's log.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from loguru import logger

from runtimev2.workers.loop import run_at
from utils.multiprocessing import ensure_spawn_start_method
from workers.base import ManagedLifecycle

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from contextlib import AbstractContextManager

_READY_TIMEOUT_S = 30.0
"""Long enough for a port to open and every servo to answer a ping.

Bounded, unlike the thread version: a child that dies hard never reaches its
``finally``, so nothing would set the event.
"""

_JOIN_TIMEOUT_S = 5.0
"""Long enough to finish a tick and release the device before being killed."""


class ProcessWorker(mp.Process, ManagedLifecycle[None], ABC):
    """Hold a device in its own process and tick it at a fixed rate.

    Subclasses say what they hold in :meth:`acquire` and what one pass does in
    :meth:`tick`; starting, stopping, killing a wedged one and reporting a
    failure live here. Single-use, like the process it is.
    """

    def __init__(self, *, name: str, hz: float) -> None:
        ensure_spawn_start_method()
        # Daemon so a wedged worker cannot keep the parent alive; it is joined
        # and then killed on the way out.
        super().__init__(name=name, daemon=True)
        self._hz = hz
        # Created here so spawning carries them over: one handed across later
        # would not be shared.
        self._stop = mp.Event()
        self._ready = mp.Event()
        self._failed = mp.Event()
        self._failures: mp.Queue[str] = mp.Queue()

    @property
    def hz(self) -> float:
        return self._hz

    @abstractmethod
    def acquire(self) -> AbstractContextManager[None]:
        """Hold whatever this worker needs for the body of the block.

        Runs *in the child*, so this is where an unpicklable device is built.
        """

    @abstractmethod
    def tick(self) -> None:
        """Do one pass. No timing: the rate is the loop's business."""

    def run(self) -> None:
        """Hold the device and tick until stopped. The child's body.

        Failures are reported, not raised: nothing in the parent watches this.
        """
        from core.logging import setup_logging

        # Loguru sinks do not cross a process boundary; without this a child
        # reports its failures into nothing, which looks like being fine.
        setup_logging()
        try:
            with self.acquire():
                self._ready.set()
                run_at(self, self._hz, self._stop.is_set)
        except BaseException as error:  # reported to the parent, then we exit
            # Message before flag: the queue has a feeder thread, so a parent
            # that sees the flag can block for what is behind it.
            self._failures.put(f"{type(error).__name__}: {error}")
            self._failed.set()
            logger.exception("Worker {} failed", self.name)
        finally:
            # Set again in case acquiring failed: nobody should wait on a
            # process that has gone.
            self._ready.set()

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[None]:
        """Start the process, wait until it holds its device, stop it after.

        Yields only once the device is held, so a failure to acquire fails the
        load rather than going unnoticed in a child.

        Raises:
            RuntimeError: The child failed to acquire, died, or never readied.
        """
        self.start()
        await asyncio.to_thread(self._ready.wait, _READY_TIMEOUT_S)
        self._reject_failure()
        try:
            yield
        finally:
            self._stop.set()
            await asyncio.to_thread(self.join, _JOIN_TIMEOUT_S)
            if self.is_alive():
                # It is holding a device; a wedged child must not keep it.
                logger.warning("Worker {} did not stop within {}s; killing it", self.name, _JOIN_TIMEOUT_S)
                self.kill()
                await asyncio.to_thread(self.join, _JOIN_TIMEOUT_S)
            elif self._failed.is_set():
                # Died mid-run: the environment looked healthy while this
                # device went quiet.
                logger.error("Worker {} stopped early: {}", self.name, self._reason())
            self._failures.close()

    def _reject_failure(self) -> None:
        """Re-raise in the parent whatever the child failed with.

        Raises:
            RuntimeError: It failed, died, or never reported being ready.
        """
        if self._failed.is_set():
            raise RuntimeError(f"Worker {self.name} failed to start: {self._reason()}")
        if not self.is_alive() and self.exitcode not in (0, None):
            raise RuntimeError(f"Worker {self.name} exited with code {self.exitcode}")
        if not self._ready.is_set():
            raise RuntimeError(f"Worker {self.name} did not start within {_READY_TIMEOUT_S:.0f}s")

    def _reason(self) -> str:
        """The child's failure message, or a note that it did not leave one."""
        try:
            return self._failures.get(timeout=_JOIN_TIMEOUT_S)
        except Exception:  # an empty or closed queue tells us nothing more
            return "no reason reported"

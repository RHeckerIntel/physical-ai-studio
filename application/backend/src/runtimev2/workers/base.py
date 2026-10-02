"""A worker that owns its device, its thread and its rate.

A worker *is* a thread rather than something that starts one. Entering it starts
the thread and waits until the device is held; leaving it stops the thread and
waits for the device to be released. That puts the thread in the type instead of
hiding it in a helper, and it is the same shape a process needs -- swapping
``threading.Thread`` for ``multiprocessing.Process`` would leave ``acquire``,
``tick`` and the ``async with`` untouched.

Acquisition happens *on the worker's thread*, which is why :meth:`acquire` is a
plain context manager rather than an async one. Connecting a device blocks, and
this is the thread that should block on it; the caller waits on an event instead
of hopping the work somewhere else.

The rate stays out of ``tick`` itself, so a test can step a worker without a
clock and the same worker can run at 30Hz in one session and 100Hz in another.
"""

from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from loguru import logger

from runtimev2.workers.loop import run_at
from workers.base import ManagedLifecycle

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from contextlib import AbstractContextManager

# Long enough for a tick in flight to finish and the device to be released,
# short enough that unloading does not feel stuck.
_JOIN_TIMEOUT_S = 10.0


class ThreadedWorker(threading.Thread, ManagedLifecycle[None], ABC):
    """Hold a device on its own thread and tick it at a fixed rate.

    Subclasses say what they hold in :meth:`acquire` and what one pass does in
    :meth:`tick`. Everything else -- starting, waiting until the device is
    actually held, stopping, joining, and handing a failure back to whoever
    entered -- is the same for all of them and lives here.

    Single-use, like the thread it is: a worker that has been entered and left
    cannot be entered again. Build a new one, which is what loading an
    environment does anyway.
    """

    def __init__(self, *, name: str, hz: float) -> None:
        # Daemon so a wedged worker cannot keep the interpreter alive; it is
        # still joined on the way out, which is what actually waits for it.
        super().__init__(name=name, daemon=True)
        self._hz = hz
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._failure: BaseException | None = None

    @property
    def hz(self) -> float:
        return self._hz

    @abstractmethod
    def acquire(self) -> AbstractContextManager[None]:
        """Hold whatever this worker needs for the body of the block.

        Runs on the worker's own thread, before any tick and until after the
        last one, so ``tick`` can assume the device is there and blocking here
        costs the caller nothing.
        """

    @abstractmethod
    def tick(self) -> None:
        """Do one pass. No timing: the rate is the loop's business."""

    def run(self) -> None:
        """Hold the device and tick until stopped. The thread's body.

        A failure is kept rather than raised: nothing is watching this thread's
        stack, so it is handed to whoever entered the worker instead.
        """
        try:
            with self.acquire():
                self._ready.set()
                run_at(self, self._hz, self._stop.is_set)
        except BaseException as exc:
            self._failure = exc
        finally:
            # Set again in case acquiring failed: a caller waiting to be told
            # the device is held must not wait on a thread that has gone.
            self._ready.set()

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[None]:
        """Start the thread, wait until it holds its device, and stop it after.

        Yields only once the device is actually held, so a caller that gets past
        this knows the worker is live -- and a failure to acquire surfaces here,
        where it can fail the load, rather than on a thread nobody is watching.

        Yields nothing: whoever enters a worker already holds it.

        Raises:
            BaseException: Whatever ``acquire`` raised on the worker's thread.
        """
        self.start()
        await asyncio.to_thread(self._ready.wait)
        if self._failure is not None:
            await asyncio.to_thread(self.join, _JOIN_TIMEOUT_S)
            raise self._failure
        try:
            yield
        finally:
            self._stop.set()
            await asyncio.to_thread(self.join, _JOIN_TIMEOUT_S)
            if self.is_alive():
                logger.warning("Worker {} did not stop within {}s", self.name, _JOIN_TIMEOUT_S)
            elif self._failure is not None:
                # It died mid-run rather than failing to start. Worth saying so:
                # the environment looked healthy while this device went quiet.
                logger.error("Worker {} stopped early: {}", self.name, self._failure)

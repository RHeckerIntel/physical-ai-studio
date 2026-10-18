import asyncio
import contextlib
import multiprocessing as mp
import os
from contextlib import AsyncExitStack

import psutil
from loguru import logger

from workers.dataset_import_worker import DatasetImportWorker
from workers.training_worker import TrainingWorker


class Scheduler:
    """Manages application processes and threads."""

    def __init__(self) -> None:
        logger.info("Initializing Scheduler...")
        # Shared, per-job interrupt flags (job id -> True) so cancelling one
        # training job cannot cross-cancel another job running concurrently
        # on a different target. Backed by a Manager dict since training runs
        # in a separate process from the API.
        self.manager = mp.Manager()
        self.job_interrupt_flags = self.manager.dict()
        self.event_queue: mp.Queue = mp.Queue()

        self._workers = [
            TrainingWorker(job_interrupt_flags=self.job_interrupt_flags, event_queue=self.event_queue),
            DatasetImportWorker(event_queue=self.event_queue),
        ]
        self._stack = AsyncExitStack()
        self._starting: asyncio.Task[None] | None = None
        logger.info("Scheduler initialized")

    async def start_workers(self) -> None:
        """Start the workers in the background and return straight away.

        A worker is ready once its child has imported its modules (torch, for training),
        which takes seconds. Nothing needs them before then: jobs wait in the database.
        """
        self._starting = asyncio.create_task(self._start_workers())

    async def _start_workers(self) -> None:
        for worker in self._workers:
            try:
                await self._stack.enter_async_context(worker)
            except Exception:
                # The API still serves without a background worker; started in the background, nothing else reports it.
                logger.exception("Background worker {} failed to start", worker.name)

    async def shutdown(self) -> None:
        """Shutdown all processes gracefully"""
        logger.info("Initiating graceful shutdown...")
        pid = os.getpid()
        alive_children = [child.pid for child in psutil.Process(pid).children(recursive=True) if child.is_running()]
        logger.debug(f"Alive children of process '{pid}': {alive_children}")

        if self._starting is not None and not self._starting.done():
            # Cancelling a worker mid-start stops and joins it, so nothing is left behind.
            self._starting.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._starting
        # Ask every worker first so they wind down in parallel; the stack then
        # waits for each (killing any that hang) in reverse start order.
        for worker in self._workers:
            worker.request_stop()
        await self._stack.aclose()
        logger.info("All workers shut down gracefully")
        self.manager.shutdown()

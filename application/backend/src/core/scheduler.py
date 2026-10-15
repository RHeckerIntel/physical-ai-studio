import multiprocessing as mp
import os
from contextlib import AsyncExitStack

import psutil
from loguru import logger

from workers.base import WorkerStartupError
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
        logger.info("Scheduler initialized")

    async def start_workers(self) -> None:
        for worker in self._workers:
            try:
                await self._stack.enter_async_context(worker)
            except WorkerStartupError:
                # As before: the API still serves without a background worker.
                logger.exception("Background worker {} failed to start", worker.name)

    async def shutdown(self) -> None:
        """Shutdown all processes gracefully"""
        logger.info("Initiating graceful shutdown...")
        pid = os.getpid()
        alive_children = [child.pid for child in psutil.Process(pid).children(recursive=True) if child.is_running()]
        logger.debug(f"Alive children of process '{pid}': {alive_children}")

        # Ask every worker first so they wind down in parallel; the stack then
        # waits for each (killing any that hang) in reverse start order.
        for worker in self._workers:
            worker.request_stop()
        await self._stack.aclose()
        logger.info("All workers shut down gracefully")
        self.manager.shutdown()

import multiprocessing as mp
import os
from typing import TYPE_CHECKING

import psutil
from loguru import logger

from workers.base import BaseProcessWorker
from workers.dataset_import_worker import DatasetImportWorker
from workers.training_worker import TrainingWorker

if TYPE_CHECKING:
    import threading


class Scheduler:
    """Manages application processes and threads."""

    def __init__(self) -> None:
        logger.info("Initializing Scheduler...")
        # Event to sync all processes on application shutdown
        self.mp_stop_event = mp.Event()
        # Shared, per-job interrupt flags (job id -> True) so cancelling one
        # training job cannot cross-cancel another job running concurrently
        # on a different target. Backed by a Manager dict since training runs
        # in a separate process from the API.
        self.manager = mp.Manager()
        self.job_interrupt_flags = self.manager.dict()
        self.event_queue: mp.Queue = mp.Queue()

        self.processes: list[BaseProcessWorker] = []
        self.threads: list[threading.Thread] = []
        logger.info("Scheduler initialized")

    def start_workers(self) -> None:
        # mp.set_start_method("spawn", force=True)
        training_proc = TrainingWorker(
            stop_event=self.mp_stop_event,
            job_interrupt_flags=self.job_interrupt_flags,
            event_queue=self.event_queue,
        )
        training_proc.daemon = False
        training_proc.start()

        dataset_import_proc = DatasetImportWorker(
            stop_event=self.mp_stop_event,
            event_queue=self.event_queue,
        )
        dataset_import_proc.daemon = False
        dataset_import_proc.start()

        self.processes.extend([training_proc, dataset_import_proc])

    def shutdown(self) -> None:
        """Shutdown all processes gracefully"""
        logger.info("Initiating graceful shutdown...")

        # Signal all processes to stop
        self.mp_stop_event.set()

        # Get current process info for debugging
        pid = os.getpid()
        cur_process = psutil.Process(pid)
        alive_children = [child.pid for child in cur_process.children(recursive=True) if child.is_running()]
        logger.debug(f"Alive children of process '{pid}': {alive_children}")

        # Join threads first
        for thread in self.threads:
            if thread.is_alive():
                logger.debug(f"Joining thread: {thread.name}")
                thread.join(timeout=10)
                if thread.is_alive():
                    logger.warning(f"Thread {thread.name} did not terminate within timeout")

        # Reverse order so that consumers stop before the producers they read from.
        # ``stop()`` owns the join/SIGTERM/SIGKILL escalation; restating it here
        # would be a second copy to keep in step with the worker's own teardown.
        for process in self.processes[::-1]:
            logger.debug(f"Stopping process: {process.name}")
            process.stop()

        logger.info("All workers shut down gracefully")

        # Clear references
        self.processes.clear()
        self.threads.clear()
        self.manager.shutdown()

from __future__ import annotations

import abc
import asyncio
import multiprocessing as mp
import os
import signal
import threading
import time
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack, asynccontextmanager
from multiprocessing import Event
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Iterable
    from contextlib import AbstractAsyncContextManager
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Event as EventClass
    from types import FrameType
    from typing import Self

import loguru
from loguru import logger

# Escalation budget shared by every process worker: a graceful stop first, then
# SIGTERM, then SIGKILL. Callers that need longer for teardown pass their own
# ``timeout``; the two grace periods after it are not worth tuning per caller.
DEFAULT_STOP_TIMEOUT_S = 10.0
_TERMINATE_GRACE_S = 2.0
_KILL_GRACE_S = 1.0
_JOIN_POLL_S = 0.1


@asynccontextmanager
async def run_at_frequency(frequency: float) -> AsyncGenerator[None]:
    """Run a function at a specified frequency.

    Note: This is a frequency, not time (1/s vs s).
    Example:

    fps = 30
    async with run_at_frequency(fps):
       get_frame_from_camera()
    """

    t0 = time.perf_counter()
    yield
    target_dt = 1 / frequency
    elapsed = time.perf_counter() - t0
    sleep_time = target_dt - elapsed
    if sleep_time > 0:
        await asyncio.sleep(sleep_time)
    else:
        logger.debug(f"Missed timing on run_at_frequency: {-sleep_time * 1000}ms")
        await asyncio.sleep(0)


def log_threads(log_level="DEBUG") -> None:  # noqa: ANN001
    """Log all the alive threads associated with the current process"""
    pid = os.getpid()
    alive_threads = [thread for thread in threading.enumerate() if thread.is_alive()]
    thread_list_msg = (
        f"Alive threads for process with pid '{pid}': "
        f"{', '.join([str((thread.name, thread.ident)) for thread in alive_threads])}"
    )
    logger.log(log_level, thread_list_msg)


class ManagedLifecycle[Context](ABC):
    """Separate what a subclass holds from the guarantee that it is released.

    A subclass implements :meth:`lifecycle`, acquiring what it needs and
    releasing it in the same place. This class turns that into an ``async
    with`` and takes on the two parts every caller would otherwise repeat:

    * Partial acquisition. The stack unwinds only what was entered, in reverse,
      so a resource is released before whatever it was built on. That covers
      ``lifecycle`` raising midway, where ``__aexit__`` is never called and the
      caller has no handle to clean up with.
    * Cancellation, including a second one arriving while the first is still
      unwinding. An unshielded unwind stops at its first ``await`` and leaves
      the rest held, so the unwind runs as a shielded task instead.

    ``Context`` is whatever :meth:`lifecycle` yields, and is what ``async
    with`` binds.
    """

    _lifecycle_stack: AsyncExitStack | None = None

    @abstractmethod
    def lifecycle(self) -> AbstractAsyncContextManager[Context]:
        """Acquire everything this needs, yield it, then release it."""

    async def __aenter__(self) -> Context:
        stack = AsyncExitStack()
        self._lifecycle_stack = stack
        try:
            return await stack.enter_async_context(self.lifecycle())
        except BaseException:
            # __aexit__ does not run for a failed __aenter__, so whatever the
            # lifecycle acquired before it raised is released here instead.
            self._lifecycle_stack = None
            await asyncio.shield(asyncio.create_task(stack.aclose()))
            raise

    async def __aexit__(self, *args: object) -> None:
        stack, self._lifecycle_stack = self._lifecycle_stack, None
        if stack is not None:
            await asyncio.shield(asyncio.create_task(stack.aclose()))


class StoppableMixin:
    """Mixin providing stop-aware functionality using external stop event."""

    # Set from a signal handler, so it stays a plain attribute rather than an
    # Event: setting an Event takes its lock, and taking that lock inside a
    # handler can deadlock against the code the signal interrupted.
    _signalled: bool = False

    def should_stop(self) -> bool:
        """Check if a stop has been requested."""
        if not hasattr(self, "_interrupt_event"):
            raise AttributeError("StoppableMixin requires a '_interrupt_event' to be set.")

        if not hasattr(self, "_stop_event"):
            raise AttributeError("StoppableMixin requires a '_stop_event' to be set.")

        # Stop if parent process died
        parent_process = mp.parent_process()
        parent_died = parent_process is not None and not parent_process.is_alive()

        return self._signalled or self._interrupt_event.is_set() or self._stop_event.is_set() or parent_died  # type: ignore

    def stop_aware_sleep(self, seconds: float) -> bool:
        """
        Sleep for the specified time, but wake up immediately if stop is requested.

        Args:
            seconds: Maximum time to sleep in seconds

        Returns:
            True if woke up due to stop request, False if timeout elapsed
        """
        if not hasattr(self, "_stop_event"):
            raise AttributeError("StoppableMixin requires _stop_event to be set")
        return self._stop_event.wait(seconds)  # type: ignore


class BaseProcessWorker(mp.Process, StoppableMixin, ABC):
    """
    Reusable worker with a clean lifecycle: setup() -> run_loop() [until stop_event] -> teardown()
    Subclasses only implement what's specific to their job.
    """

    # Override in subclasses for a nicer auto-name:
    ROLE: str = "Worker"

    loop: asyncio.AbstractEventLoop | None = None

    def __init__(
        self,
        *,
        stop_event: EventClass,
        queues_to_cancel: Iterable[Queue] | None = None,
        logger_: loguru.Logger | None = None,
    ) -> None:
        super().__init__()
        self._interrupt_event = stop_event
        self._stop_event = Event()
        self._parent_pid = os.getpid()
        self._queues_to_cancel = list(queues_to_cancel or [])

        # Platforms that use "spawn" for multiprocessing (e.g. Windows) cause logging concurrency issues.
        # Therefore, we need to copy the logger with enqueue=True in child processes.
        # https://loguru.readthedocs.io/en/stable/resources/recipes.html#compatibility-with-multiprocessing-using-enqueue-argument
        global logger  # noqa: PLW0603
        logger = logger_ or logger

    # Hooks to be implemented by subclasses

    async def setup(self) -> None:
        """Allocate resources and initialize settings. Called once in the child process."""
        # Logging must be re-setup in child processes because loguru sinks don't
        # transfer across process boundaries and settings are non-picklable.
        from core.logging import setup_logging

        setup_logging()

    @abstractmethod
    async def run_loop(self) -> None:
        """
        Main loop. Return only when asked to stop or on unrecoverable error.
        self.should_stop() should be used the loop condition.
        """
        ...

    async def teardown(self) -> None:
        """Release resources (optional)."""

    # Internal + final run orchestration

    def _install_signal_policy(self) -> None:
        """
        Route shutdown signals in child processes through the stop flag.

        SIGINT stays ignored: Ctrl+C reaches the whole process group, and cleanup is
        coordinated through the parent process via the stop_event mechanism.

        SIGTERM is handled rather than left on its default disposition, which ends the
        process between two bytecodes and so skips ``teardown()``. A worker that dies
        that way strands whatever it holds -- a camera publisher, for one, infers that
        its last subscriber is gone from a clean disconnect, and without one it keeps
        the device open indefinitely.
        """
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, self._handle_terminate)

    def _handle_terminate(self, _signum: int, _frame: FrameType | None) -> None:
        """Ask the run loop to stop so ``run()`` still reaches ``teardown()``."""
        self._signalled = True

    def _auto_name(self) -> str:
        """Generate a name for the process based on its role and PIDs."""
        return "-".join([self.ROLE, str(self._parent_pid), str(os.getpid())])

    def _cancel_queue_join_threads(self) -> None:
        for q in self._queues_to_cancel:
            try:
                # https://docs.python.org/3/library/multiprocessing.html#all-start-methods
                # section: Joining processes that use queues
                # Call cancel_join_thread() to prevent the parent process from blocking
                # indefinitely when joining child processes that used this queue. This avoids potential
                # deadlocks if the queue's background thread adds more items during the flush.
                q.cancel_join_thread()
                logger.debug(f"Cancelled join thread for queue {getattr(q, 'name', q)!r}")
            except Exception as e:
                logger.warning(f"Failed cancelling queue join thread: {e}")

    def run(self) -> None:
        with logger.contextualize(worker=self.__class__.__name__):
            self._install_signal_policy()
            self.name = self._auto_name()
            logger.info(f"Starting {self.name}...")

            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            try:
                self.loop.run_until_complete(self.setup())
                self.loop.run_until_complete(self.run_loop())
            except Exception:
                logger.exception(f"Unhandled exception in {self.name}")
            finally:
                try:
                    self.loop.run_until_complete(self.teardown())
                finally:
                    self.loop.run_until_complete(self.loop.shutdown_asyncgens())
                    self.loop.close()

                    self._cancel_queue_join_threads()
                    log_threads()
                    logger.info(f"Stopped {self.name}.")

    def request_stop(self) -> None:
        """Ask the worker to stop without waiting for it."""
        self._stop_event.set()

    def stop(
        self,
        timeout: float = DEFAULT_STOP_TIMEOUT_S,
        *,
        on_poll: Callable[[], None] | None = None,
    ) -> None:
        """Stop the worker and wait for it to exit. Blocking, idempotent.

        Graceful first: the worker sees the request through its stop event and runs
        ``teardown()``, releasing whatever it holds. Only a worker that does not
        exit within *timeout* is escalated to SIGTERM and then SIGKILL -- SIGTERM
        still reaches ``teardown()`` (see ``_install_signal_policy``), SIGKILL does
        not, so anything it was holding is stranded.

        Args:
            timeout: Seconds to allow for a graceful exit before escalating. Pass
                more than the default when teardown has real work to do, such as
                copying a recording back.
            on_poll: Called between join attempts. A worker that writes to a
                ``Queue`` the parent is reading cannot exit while its feeder
                thread blocks on a full pipe, so the parent has to keep draining
                while it waits -- this is the hook for that.
        """
        self.request_stop()
        if self.pid is None:
            return

        def poll() -> None:
            if on_poll is not None:
                on_poll()

        deadline = time.monotonic() + timeout
        while self.is_alive() and time.monotonic() < deadline:
            poll()
            self.join(_JOIN_POLL_S)

        if self.is_alive():
            logger.warning(f"Process {self.name} did not stop within {timeout}s, terminating")
            self.terminate()
            self.join(timeout=_TERMINATE_GRACE_S)
            if self.is_alive():
                logger.error(f"Killing process {self.name}")
                self.kill()
                self.join(timeout=_KILL_GRACE_S)
        poll()


class BaseThreadWorker(threading.Thread, StoppableMixin, abc.ABC):
    ROLE: str = "Worker"

    def __init__(self, *, stop_event: EventClass, daemon: bool = False):
        super().__init__(daemon=daemon)
        self._interrupt_event = stop_event
        self._stop_event = Event()
        self.name = f"{self.ROLE}-{os.getpid()}-thread"
        self.loop: asyncio.AbstractEventLoop | None = None

    # hooks
    def setup(self) -> None:
        pass

    @abstractmethod
    async def run_loop(self) -> None: ...

    async def teardown(self) -> None:
        pass

    # final run orchestration
    def run(self) -> None:
        with logger.contextualize(worker=self.__class__.__name__):
            logger.info(f"Starting {self.name}")
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            try:
                self.setup()
                self.loop.run_until_complete(self.run_loop())
            except Exception:
                logger.exception(f"Unhandled exception in {self.name}")
            finally:
                try:
                    self.loop.run_until_complete(self.teardown())
                finally:
                    self.loop.run_until_complete(self.loop.shutdown_asyncgens())
                    self.loop.close()
                    log_threads()
                    logger.info(f"Stopped {self.name}")

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=DEFAULT_STOP_TIMEOUT_S)

    def __enter__(self) -> Self:
        """Start the worker, stopping it when the block exits.

        The worker holds a device for as long as it runs, so the ``stop()`` that
        releases it belongs to the scope that started it rather than to a
        ``finally`` each caller has to remember.
        """
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

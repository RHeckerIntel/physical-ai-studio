from contextlib import AbstractAsyncContextManager, asynccontextmanager
from abc import abstractmethod
import asyncio
from workers.base import StoppableMixin
import threading
import multiprocessing as mp
from multiprocessing.synchronize import Event as EventClass
from loguru import logger
from typing import Self, Any,AsyncGenerator

class BaseThreadWorker[Context](threading.Thread, StoppableMixin):
    def __init__(self,  stop_event: EventClass):
        super().__init__()
        self._interrupt_event = stop_event
        self._stop_event = mp.Event()
        self._loaded_event = mp.Event()

    @abstractmethod
    def lifecycle(self) -> AbstractAsyncContextManager[Context]:
        """Acquire everything, yield to run loop and then clean up."""

    @abstractmethod
    async def run_loop(self, context: Context) -> None:
        """Run loop of process."""

    def run(self) -> None:
        with logger.contextualize(worker=self.__class__.__name__):
            try:
                logger.info(f"Starting {self.name}")
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                async def _main() -> None:
                    async with self.lifecycle() as ctx:
                        self._loaded_event.set()
                        await self.run_loop(ctx)
                self.loop.run_until_complete(_main())
                logger.info(f"Stopped {self.name}")
            except Exception:
                logger.exception(f"Unhandled exception in {self.name}")
            finally:
                self.loop.run_until_complete(self.loop.shutdown_asyncgens())
                self.loop.close()
                self._loaded_event.set() # release waiters for readiness
                logger.info(f"Stopped {self.name}.")

    async def wait_for_loaded(self):
        await asyncio.to_thread(self._loaded_event.wait)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=10)

    @classmethod
    @asynccontextmanager
    async def in_context_manager(
        cls, *args: Any, stop_event: EventClass, **kwargs: Any
    ) -> AsyncGenerator[Self]:
        instance = cls(*args, stop_event=stop_event, **kwargs)
        instance.start()
        await instance.wait_for_loaded()
        yield instance
        instance.stop()

from contextlib import AbstractAsyncContextManager
from abc import abstractmethod
import asyncio
from workers.base import StoppableMixin
import threading
import multiprocessing as mp
from multiprocessing.synchronize import Event as EventClass
from loguru import logger

class BaseThreadWorker[Context](threading.Thread, StoppableMixin):
    def __init__(self,  stop_event: EventClass):
        super().__init__()
        self._interrupt_event = stop_event
        self._stop_event = mp.Event()

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
                        await self.run_loop(ctx)
                self.loop.run_until_complete(_main())
                logger.info(f"Stopped {self.name}")
            except Exception:
                logger.exception(f"Unhandled exception in {self.name}")
            finally:
                self.loop.run_until_complete(self.loop.shutdown_asyncgens())
                self.loop.close()
                logger.info(f"Stopped {self.name}.")

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=10)

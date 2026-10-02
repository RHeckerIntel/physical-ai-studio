"""A client's runtime session: at most one loaded environment at a time.

The session is the connection, not the hardware. It holds no robots and no
store of its own -- it loads an environment, which owns those, and can unload
it and load another without the client reconnecting. Swapping is the point:
which environment is loaded decides the feature spec, and therefore which
datasets and models fit.

Loading is exclusive rather than additive. Two environments sharing a robot
would both hold the same device, so the previous one is unloaded first and its
arms are released before the next one claims anything.
"""

from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from runtimev2.loaded_environment import DEFAULT_ROBOT_HZ, LoadedEnvironment
from workers.base import ManagedLifecycle

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from robots.robot_client_factory import RobotClientFactory
    from runtimev2.loaded_environment import EnvironmentState
    from schemas.environment import EnvironmentWithRelations


@dataclass(frozen=True, slots=True)
class SessionState:
    """What the session looks like from outside.

    ``environment`` is ``None`` when nothing is loaded, which is the state a
    session starts in and returns to after an unload.
    """

    loaded: bool
    environment: EnvironmentState | None = None


class RuntimeSession(ManagedLifecycle["RuntimeSession"]):
    """One client connection, holding at most one loaded environment.

    Opened with ``async with``; leaving unloads whatever is still loaded.
    """

    def __init__(self, factory: RobotClientFactory, *, robot_hz: float = DEFAULT_ROBOT_HZ) -> None:
        self._factory = factory
        self._robot_hz = robot_hz
        self._loaded: LoadedEnvironment | None = None
        # The loaded environment's own teardown, kept so it can be closed on its
        # own rather than only when the session ends.
        self._stack: AsyncExitStack | None = None

    @property
    def environment(self) -> LoadedEnvironment | None:
        """The loaded environment, or ``None``."""
        return self._loaded

    def require_environment(self) -> LoadedEnvironment:
        """Return the loaded environment.

        Raises:
            RuntimeError: Nothing is loaded.
        """
        if self._loaded is None:
            raise RuntimeError("No environment is loaded")
        return self._loaded

    def state(self) -> SessionState:
        return SessionState(
            loaded=self._loaded is not None,
            environment=self._loaded.state() if self._loaded is not None else None,
        )

    async def load(self, environment: EnvironmentWithRelations) -> LoadedEnvironment:
        """Load ``environment``, unloading whatever was loaded before.

        A failed load leaves the session empty rather than holding a half-built
        environment: whatever was acquired before the failure is released, and
        the previous environment is already gone by then -- it is unloaded
        first so its robots are free for this one to claim.
        """
        await self.unload()
        stack = AsyncExitStack()
        try:
            loaded = await stack.enter_async_context(
                LoadedEnvironment(environment, self._factory, robot_hz=self._robot_hz)
            )
        except BaseException:
            await stack.aclose()
            raise
        self._loaded, self._stack = loaded, stack
        return loaded

    async def unload(self) -> None:
        """Release the loaded environment. Idempotent."""
        stack, self._stack = self._stack, None
        loaded, self._loaded = self._loaded, None
        if stack is None:
            return
        # Cleared before closing, so a client asking mid-unload is told nothing
        # is loaded rather than handed devices on their way out.
        logger.info("Unloading {}", loaded.name if loaded else "environment")
        await stack.aclose()

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[RuntimeSession]:
        """Hold the session open, unloading anything still loaded on the way out."""
        try:
            yield self
        finally:
            await self.unload()

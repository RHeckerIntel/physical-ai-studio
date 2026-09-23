from typing import Any
from contextlib import AsyncExitStack
import asyncio
from uuid import UUID, uuid4
from runtimev2.session import RuntimeSession, runtime_session

class SessionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[UUID, RuntimeSession] = {}
        self._stacks: dict[UUID, AsyncExitStack] = {}
        self._lock = asyncio.Lock()

    async def open(self, cfg: dict[str, Any]) -> RuntimeSession:
        async with self._lock:
            stack = AsyncExitStack()
            try:
                session = await stack.enter_async_context(runtime_session(cfg))
            except BaseException:
                await stack.aclose()
                raise
            sid = uuid4()
            self._sessions[sid], self._stacks[sid] = session, stack
            return session

    async def close(self, sid: UUID) -> None:
        async with self._lock:
            stack = self._stacks.pop(sid, None)
            self._sessions.pop(sid, None)
        if stack is not None:
            await stack.aclose()

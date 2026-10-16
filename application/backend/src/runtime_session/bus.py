"""Commands, events and requests between the processes of one session, over zenoh.

The session process :meth:`Bus.host`\\ s a zenoh peer listening on a Unix
socket; its worker processes :meth:`Bus.join` by connecting to that socket.
Nothing listens on the network and sessions cannot hear each other. Keys are
relative to ``pai/<session_id>/``: ``cmd/<component>`` and ``evt/<component>``
for pub/sub, ``rpc/<component>/<op>`` for requests. Payloads are pydantic
models as JSON. Joints and frames go through the :mod:`store`, not the bus.

See the "Event bus" section of docs/feature_plans/runtime-inhousing-plan.md.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import zenoh
from loguru import logger
from pydantic import BaseModel, TypeAdapter

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_ROOT = "pai"


class BusRequestError(RuntimeError):
    """The handler of a request raised; carries its message."""


def socket_path(session_id: str) -> Path:
    """Where the hosting session listens."""
    # shortcut: a fixed temp-dir path assumes one backend per host (like store blocks); use a per-backend dir otherwise.
    return Path(tempfile.gettempdir()) / f"pai-{session_id}.sock"


class Declaration:
    """A subscriber or queryable on the bus; close it, or use it as a context manager, to undeclare it.

    Whatever is still open when the bus closes is undeclared with it.
    """

    def __init__(self, open_declarations: set[Declaration]) -> None:
        self._entity: zenoh.Subscriber | zenoh.Queryable | None = None
        self._open = open_declarations
        self._open.add(self)

    def close(self) -> None:
        self._open.discard(self)
        if self._entity is not None:
            self._entity.undeclare()
            self._entity = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class Subscription[M](Declaration):
    """Messages on one key, handed to the event loop that subscribed."""

    def __init__(self, model: Any, loop: asyncio.AbstractEventLoop, open_declarations: set[Declaration]) -> None:
        super().__init__(open_declarations)
        self._adapter: TypeAdapter[M] = TypeAdapter(model)
        self._loop = loop
        self._queue: asyncio.Queue[M] = asyncio.Queue()

    def _on_sample(self, sample: zenoh.Sample) -> None:  # zenoh thread
        try:
            message = self._adapter.validate_json(sample.payload.to_bytes())
        except ValueError:
            logger.exception("Dropped a malformed message on {}", sample.key_expr)
            return
        self._loop.call_soon_threadsafe(self._queue.put_nowait, message)

    async def next(self) -> M:
        return await self._queue.get()

    def poll(self) -> M | None:
        """The oldest undelivered message, or ``None``; for fixed-rate loops."""
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None


class Bus:
    """One process's zenoh session, scoped to a runtime session."""

    def __init__(self, session_id: str, config: zenoh.Config) -> None:
        self.session_id = session_id
        self._prefix = f"{_ROOT}/{session_id}/"
        self._session = zenoh.open(config)
        self._declared: set[Declaration] = set()

    @classmethod
    def host(cls, session_id: str) -> Bus:
        """The session process's bus: listens on the session's socket."""
        path = socket_path(session_id)
        path.unlink(missing_ok=True)  # left behind by a killed session
        return cls(session_id, _config(listen=path))

    @classmethod
    def join(cls, session_id: str) -> Bus:
        """A worker process's bus: connects to the hosting session."""
        return cls(session_id, _config(connect=socket_path(session_id)))

    def publish(self, key: str, message: BaseModel) -> None:
        self._session.put(self._prefix + key, message.model_dump_json())

    def subscribe[M](self, key: str, model: type[M] | Any) -> Subscription[M]:
        """Receive messages on ``key``; call from the event loop that will consume them.

        ``model`` is a pydantic model or a discriminated union of them. Messages
        published before the subscription has reached the publisher's peer are
        lost; use :meth:`request` when delivery must be confirmed.
        """
        subscription: Subscription[M] = Subscription(model, asyncio.get_running_loop(), self._declared)
        subscription._entity = self._session.declare_subscriber(self._prefix + key, subscription._on_sample)
        return subscription

    async def request[R](self, key: str, message: BaseModel, response: type[R] | Any, *, timeout: float) -> R:  # noqa: ASYNC109 - zenoh enforces it
        """Send ``message`` to the handler serving ``key`` and return its reply.

        Raises:
            BusRequestError: The handler raised.
            TimeoutError: Nobody serves ``key`` or the handler did not reply in time.
        """
        replies = await asyncio.to_thread(self._get, self._prefix + key, message.model_dump_json(), timeout)
        # zenoh reports its own timeout as an error reply without a replier
        replies = [reply for reply in replies if reply.ok is not None or reply.replier_id is not None]
        if not replies:
            raise TimeoutError(f"No reply on {key} within {timeout}s")
        reply = replies[0]
        if reply.ok is None:
            raise BusRequestError(reply.err.payload.to_string() if reply.err else "")
        adapter: TypeAdapter[R] = TypeAdapter(response)
        return adapter.validate_json(reply.ok.payload.to_bytes())

    def _get(self, key: str, payload: str, timeout: float) -> list[zenoh.Reply]:
        return list(self._session.get(key, payload=payload, timeout=timeout))

    def serve[M](
        self, key: str, model: type[M] | Any, handler: Callable[[M], BaseModel | Awaitable[BaseModel]]
    ) -> Declaration:
        """Answer requests on ``key``; call from the event loop async handlers run on.

        An exception in ``handler`` is sent back and raised in the requester as
        :class:`BusRequestError`.
        """
        loop = asyncio.get_running_loop()
        adapter: TypeAdapter[M] = TypeAdapter(model)

        async def answer(query: zenoh.Query) -> None:
            try:
                result = handler(adapter.validate_json(query.payload.to_bytes() if query.payload else b""))
                if inspect.isawaitable(result):
                    result = await result
                query.reply(query.key_expr, result.model_dump_json())
            except Exception as error:  # every failure goes back to the requester
                logger.opt(exception=error).debug("Request on {} failed", key)
                query.reply_err(str(error))
            finally:
                query.drop()

        def on_query(query: zenoh.Query) -> None:  # zenoh thread
            asyncio.run_coroutine_threadsafe(answer(query), loop)

        served = Declaration(self._declared)
        served._entity = self._session.declare_queryable(self._prefix + key, on_query)
        return served

    def close(self) -> None:
        for declared in list(self._declared):
            declared.close()
        self._session.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _config(*, listen: Path | None = None, connect: Path | None = None) -> zenoh.Config:
    config = zenoh.Config()
    config.insert_json5("mode", '"peer"')
    config.insert_json5("scouting/multicast/enabled", "false")
    config.insert_json5("listen/endpoints", json.dumps([f"unixsock-stream/{listen}"] if listen else []))
    config.insert_json5("connect/endpoints", json.dumps([f"unixsock-stream/{connect}"] if connect else []))
    return config

"""What drives the follower: one control at a time, writing ``action``.

A control is a thread in the session's process, ticking at its own rate. It
borrows what the session owns (the store, the leader) and never connects or
disconnects anything, so switching controls is cheap and leaves every device
as it was. The robot worker follows whatever ``action`` holds, at its own rate.

See the "Controls" section of docs/feature_plans/runtime-inhousing-plan.md.
"""

from __future__ import annotations

from abc import abstractmethod
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from workers.base import ThreadWorker
from workers.rate import Rate

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from runtime_session.store import Store
    from schemas.runtime_session import ControlConfig


class Control(ThreadWorker[None]):
    """Calls :meth:`on_start` once, then :meth:`on_tick` at ``hz`` until stopped.

    Both run on the control's thread. An exception in ``on_start`` fails the
    start (the session then holds); one in ``on_tick`` ends the control.
    """

    def __init__(self, store: Store, config: ControlConfig, *, name: str, hz: float) -> None:
        super().__init__(name=name)
        self.store = store
        self.hz = hz
        self.config = config
        """What was asked for; the session reports it in its state."""

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[None]:
        self.on_start()
        yield

    async def run_loop(self, context: None) -> None:  # noqa: ARG002
        rate = Rate(self.hz, name=self.name)
        while not self.should_stop():
            self.on_tick()
            await rate.sleep()

    def on_start(self) -> None:
        """Capture what the control starts from, before its first tick. Does nothing by default."""

    @abstractmethod
    def on_tick(self) -> None:
        """Decide and write ``action``."""

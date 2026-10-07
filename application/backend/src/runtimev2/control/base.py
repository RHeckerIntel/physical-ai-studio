"""What a robot should be commanded to do, written to the store.

A control runs at its own rate and writes features; it does not touch a robot.
The robot reads :data:`~runtimev2.features.ACTION_KEY` at its own rate and
interpolates toward it, so the two rates are independent and neither has to
know the other.

A control writes whatever it decides to -- usually the action, but a control
with more to say can publish alongside it, which a single returned value could
not express. Both the state and the action always exist, so a control need not
be told which robot it is acting on; it reads where the arm is and writes where
it should be.

Exactly one control writes at a time. That is enforced by there being one
control slot, not by arbitration: two writers on the same joints would race.
Anything wanting both inputs -- a human correcting a policy -- is one control
holding two of them and choosing between them.
"""

from __future__ import annotations

from abc import ABC
from contextlib import contextmanager
from typing import TYPE_CHECKING

from runtimev2.workers.base import ThreadedWorker

if TYPE_CHECKING:
    from collections.abc import Generator

    from runtimev2.session_store import SessionStore


class ControlAlgorithm(ThreadedWorker, ABC):
    """A control loop: decide at its own rate, write what it decided.

    Subclasses implement :meth:`tick`, as any worker does. There is no separate
    "compute and return" step, so a control is free to write one feature or
    several.
    """

    def __init__(self, store: SessionStore, *, name: str, hz: float) -> None:
        super().__init__(name=name, hz=hz)
        self._store = store

    @property
    def store(self) -> SessionStore:
        """Where this control reads the arm and writes its commands."""
        return self._store

    @contextmanager
    def acquire(self) -> Generator[None]:
        """Nothing to hold. Overridden by controls that do."""
        yield

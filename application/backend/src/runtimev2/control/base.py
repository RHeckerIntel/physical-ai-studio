"""What a robot should be commanded to do, written to the store as features.

A control runs at its own rate and writes action features; it does not touch a
robot. The robot reads those features at its own rate and interpolates toward
them, so the two rates are independent and neither has to know the other.

Exactly one control writes a given robot's actions at a time. That is enforced
by there being one control slot, not by arbitration: two writers on the same
joints would race. Anything wanting both inputs -- a human correcting a policy
-- is one control holding two of them and choosing between them.

Writing the action rather than returning it is what keeps the store the record:
what a control wanted is visible and recordable whether or not a robot was
listening, and whether or not driving was enabled.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from runtimev2.workers.base import ThreadedWorker

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    import numpy as np

    from runtimev2.session_store import SessionStore


@dataclass(frozen=True, slots=True)
class ControlAction:
    """What to command, and the age of the data behind it.

    Attributes:
        values: One value per joint, in the robot's joint order.
        timestamp: When the data behind this was measured, not when it was
            computed. The gap between consecutive ones is the control's own
            period, which is what the robot interpolates across.
    """

    values: np.ndarray
    timestamp: float


class ControlAlgorithm(ThreadedWorker, ABC):
    """A control loop: compute an action at its own rate, publish it.

    Subclasses implement :meth:`compute`. Publishing is handled here so there
    is one place that writes action features.
    """

    def __init__(
        self,
        store: SessionStore,
        action_keys: Sequence[str],
        *,
        name: str,
        hz: float,
    ) -> None:
        super().__init__(name=name, hz=hz)
        self._store = store
        self._action_keys = list(action_keys)

    @property
    def action_keys(self) -> tuple[str, ...]:
        """The features this control writes. One control owns them."""
        return tuple(self._action_keys)

    @abstractmethod
    def compute(self) -> ControlAction | None:
        """The action to command now, or ``None`` if there is nothing yet.

        ``None`` is normal -- a policy whose first inference is still running
        -- and leaves the last published action standing.
        """

    @contextmanager
    def acquire(self) -> Generator[None]:
        """Nothing to hold. Overridden by controls that do."""
        yield

    def tick(self) -> None:
        """Compute an action and publish it.

        Raises:
            RuntimeError: ``compute`` returned the wrong number of values.
        """
        action = self.compute()
        if action is None:
            return
        if len(action.values) != len(self._action_keys):
            raise RuntimeError(
                f"Control {self.name} produced {len(action.values)} values for {len(self._action_keys)} action features"
            )
        self._store.write_many(
            {key: float(value) for key, value in zip(self._action_keys, action.values, strict=True)},
            timestamp=action.timestamp,
        )

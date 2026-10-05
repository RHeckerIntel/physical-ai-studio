"""A robot driver a spawned child can rebuild with no hardware attached."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class _Observation:
    joint_positions: np.ndarray
    timestamp: float


@dataclass
class FakeDriver:
    """Reports a fixed position and records what it was commanded.

    Instantiated in the child from a recipe, so it takes only plain values.
    """

    joint_names: list[str] = field(default_factory=list)
    position: float = 0.0
    _ticks: int = 0

    def connect(self) -> None:
        return

    def disconnect(self) -> None:
        return

    def get_observation(self) -> _Observation:
        self._ticks += 1
        return _Observation(
            joint_positions=np.full(len(self.joint_names), self.position, dtype=np.float32),
            # Advances with reads, as a real device's clock does: the gap
            # between readings is what a follower ramps across.
            timestamp=1_000.0 + self._ticks * 0.005,
        )

    def send_action(self, action: np.ndarray, **_: object) -> None:
        return

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from runtime_session.controls.base import Control
from runtime_session.store import ACTION, STATE

if TYPE_CHECKING:
    from runtime_session.store import Store
    from schemas.dataset import Episode
    from schemas.runtime_session import ReplayControlConfig


class ReplayControl(Control):
    """Command a recorded episode's actions at the rate it was recorded, then hold its last frame.

    Like teleop, it first closes the gap from where the follower is to the
    episode's first frame, over ``grace_s``, rather than jumping there. Each
    frame is stamped with when it is due, ``start + frame_index / fps``.
    """

    def __init__(self, store: Store, episode: Episode, config: ReplayControlConfig) -> None:
        if tuple(episode.action_keys) != store.spec.joint_names:
            raise ValueError(
                f"Episode {episode.episode_index} drives joints {episode.action_keys},"
                f" the follower has {list(store.spec.joint_names)}"
            )
        if not episode.actions:
            raise ValueError(f"Episode {episode.episode_index} has no frames")
        super().__init__(store, config, name="ReplayControl", hz=episode.fps)
        self.actions = np.asarray(episode.actions, dtype=np.float64)
        self.fps = episode.fps
        self.grace_s = config.grace_s
        self.start_position: np.ndarray | None = None
        self.started_at = 0.0

    def on_start(self) -> None:
        state = self.store.read(STATE)
        # Without a measured follower there is nothing to blend from; start on the first frame.
        self.start_position = state.value if state is not None else None
        self.started_at = time.monotonic()

    def on_tick(self) -> None:
        now = time.monotonic()
        elapsed = now - self.started_at
        if self.start_position is not None and elapsed < self.grace_s:
            target = self.start_position + (self.actions[0] - self.start_position) * (elapsed / self.grace_s)
            self.store.write(ACTION, target, timestamp=now)
            return
        played_from = self.started_at + (self.grace_s if self.start_position is not None else 0.0)
        frame = min(int((now - played_from) * self.fps), len(self.actions) - 1)
        self.store.write(ACTION, self.actions[frame], timestamp=played_from + frame / self.fps)

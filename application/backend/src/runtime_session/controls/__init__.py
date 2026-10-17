"""The controls a session can run, and building one from its config. See ``base.py``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from runtime_session.controls.base import Control
from runtime_session.controls.move import MoveControl
from runtime_session.controls.replay import ReplayControl
from runtime_session.controls.teleop import TeleopControl
from schemas.runtime_session import ReplayControlConfig, TeleopControlConfig

if TYPE_CHECKING:
    from physicalai.robot import Robot

    from runtime_session.store import Store
    from schemas.dataset import Episode
    from schemas.runtime_session import ControlConfig

__all__ = ["Control", "MoveControl", "ReplayControl", "TeleopControl", "build_control"]


def build_control(config: ControlConfig, store: Store, leader: Robot | None, episode: Episode | None = None) -> Control:
    """Raises ``ValueError`` when the environment cannot run ``config``; a replay needs its ``episode``."""
    if isinstance(config, ReplayControlConfig):
        if episode is None:
            raise ValueError(f"Episode {config.episode_index} is not in the dataset")
        return ReplayControl(store, episode, config)
    if isinstance(config, TeleopControlConfig):
        if leader is None:
            raise ValueError("Teleoperation needs a leader arm; this environment has none")
        return TeleopControl(store, leader, config)
    return MoveControl(store, config)

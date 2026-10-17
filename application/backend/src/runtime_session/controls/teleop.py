from __future__ import annotations

import time
from typing import TYPE_CHECKING

from robots.features import feature_names
from runtime_session.controls.base import Control
from runtime_session.store import ACTION, STATE

if TYPE_CHECKING:
    import numpy as np
    from physicalai.robot import Robot

    from runtime_session.store import Store
    from schemas.runtime_session import TeleopControlConfig


class TeleopControl(Control):
    """Command the follower whatever the leader measures, stamped with the leader's capture time.

    The leader can be anywhere when teleop starts, so for the first ``grace_s``
    the command blends from where the follower was to where the leader is now,
    rather than jumping there. It tracks the leader the whole time, so the
    follower is on it when the grace period ends.
    """

    def __init__(self, store: Store, leader: Robot, config: TeleopControlConfig) -> None:
        leader_joints = tuple(feature_names(list(leader.joint_names), include_velocities=False))
        if leader_joints != store.spec.joint_names:
            raise ValueError(
                f"The leader's joints {leader_joints} do not match the follower's {store.spec.joint_names}"
            )
        super().__init__(store, config, name="TeleopControl", hz=config.hz)
        self.leader = leader
        self.grace_s = config.grace_s
        self.start_position: np.ndarray | None = None
        self.started_at = 0.0

    def on_start(self) -> None:
        """Remember where the follower is, so the grace period blends from there."""
        state = self.store.read(STATE)
        # Without a measured follower there is nothing to blend from; follow the leader directly.
        self.start_position = state.value if state is not None else None
        self.started_at = time.monotonic()

    def on_tick(self) -> None:
        observation = self.leader.get_observation()
        target = observation.joint_positions
        elapsed = observation.timestamp - self.started_at
        if self.start_position is not None and elapsed < self.grace_s:
            target = self.start_position + (target - self.start_position) * (elapsed / self.grace_s)
        self.store.write(ACTION, target, timestamp=observation.timestamp)

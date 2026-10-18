from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from robots.features import feature_names
from runtime_session.controls.base import Control
from runtime_session.store import ACTION, STATE
from schemas.runtime_session import CommandDone, SetTeleopSlowdownCommand

if TYPE_CHECKING:
    from physicalai.robot import Robot

    from runtime_session.bus import Bus
    from runtime_session.controls.base import Commands
    from runtime_session.store import Sample, Store
    from schemas.runtime_session import TeleopControlConfig


class TeleopControl(Control):
    """Command the follower whatever the leader measures, stamped with the leader's capture time.

    The leader can be anywhere when teleop starts, so for the first ``grace_s``
    the command blends from where the follower was to where the leader is now,
    rather than jumping there. It tracks the leader the whole time, so the
    follower is on it when the grace period ends.

    With ``slowdown`` set it is relative instead: the follower moves by how
    far the leader moved since the start, divided by ``slowdown``. It
    starts where the follower is, so it needs no grace period; going back to
    absolute teleop afterwards closes the gap with one. Grippers stay absolute,
    so grasping works as usual while the arm moves slowly. ``set_teleop_slowdown``
    switches between the two while the control runs, so the follower is never
    left undriven by the switch.
    """

    def __init__(self, store: Store, leader: Robot, config: TeleopControlConfig, bus: Bus | None = None) -> None:
        leader_joints = tuple(feature_names(list(leader.joint_names), include_velocities=False))
        if leader_joints != store.spec.joint_names:
            raise ValueError(
                f"The leader's joints {leader_joints} do not match the follower's {store.spec.joint_names}"
            )
        super().__init__(store, config, name="TeleopControl", hz=config.hz, bus=bus)
        self.leader = leader
        self.grace_s = config.grace_s
        self.slowdown = config.slowdown
        # shortcut: a gripper is found by name, upgrade to the catalog declaring them when a robot names it otherwise.
        self.relative = np.array(["gripper" not in name for name in store.spec.joint_names])
        self.start_position: np.ndarray | None = None
        self.leader_start: np.ndarray | None = None
        self.started_at = 0.0

    def on_start(self) -> None:
        """Remember where the follower is, so the grace period blends from there."""
        self._anchor(self.store.read(STATE))

    def commands(self) -> Commands:
        return {"set_teleop_slowdown": (SetTeleopSlowdownCommand, self.set_slowdown)}

    def set_slowdown(self, command: SetTeleopSlowdownCommand) -> CommandDone:
        """Move on from the last command in the new mode, so the follower's path has no jump."""
        self.slowdown = command.slowdown
        self._anchor(self.store.read(ACTION) or self.store.read(STATE))
        self.changed(self.config.model_copy(update={"slowdown": command.slowdown}))
        return CommandDone()

    def _anchor(self, follower: Sample | None) -> None:
        """Start the grace period, or relative teleop, from ``follower`` and the leader as they are now."""
        # Without a measured follower there is nothing to blend from; follow the leader directly.
        self.start_position = follower.value if follower is not None else None
        self.started_at = time.monotonic()
        self.leader_start = None
        if self.slowdown is not None:
            if self.start_position is None:
                raise RuntimeError("Relative teleop needs the follower's position to move from")
            self.leader_start = self.leader.get_observation().joint_positions

    def on_tick(self) -> None:
        observation = self.leader.get_observation()
        target = observation.joint_positions
        elapsed = observation.timestamp - self.started_at
        if self.leader_start is not None and self.start_position is not None:
            relative = self.start_position + (target - self.leader_start) / self.slowdown
            target = np.where(self.relative, relative, target)
        elif self.start_position is not None and elapsed < self.grace_s:
            target = self.start_position + (target - self.start_position) * (elapsed / self.grace_s)
        self.store.write(ACTION, target, timestamp=observation.timestamp)

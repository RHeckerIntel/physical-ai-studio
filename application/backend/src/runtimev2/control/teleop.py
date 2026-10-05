"""Drive a follower from a leader's measured position.

Reads the leader device itself rather than the store, so the position and the
command derived from it come from one read on one thread. There is nothing
between them to fall out of phase.

What it read is published too, under the leader's observation features, so the
store holds the leader's position for a client to show -- and holds the same
value the command was derived from rather than a separately-timed one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from runtimev2.control.base import ControlAction, ControlAlgorithm
from runtimev2.features import OBSERVATION_PREFIX, joint_feature_key

if TYPE_CHECKING:
    from collections.abc import Sequence

    from runtimev2.leader import LeaderDevice
    from runtimev2.session_store import SessionStore


class JointMappingError(RuntimeError):
    """Raised when a leader and a follower cannot be mapped onto each other."""


def check_pairing(leader_joints: Sequence[str], follower_joints: Sequence[str]) -> None:
    """Fail unless a leader's joints correspond to a follower's, one for one.

    Positional, because the environment does not describe how they map and
    position is the only correspondence the identical-arm case has. Checked at
    build time so a mismatch fails on load, not part-way through driving.

    Raises:
        JointMappingError: The two have different joint counts.
    """
    if len(leader_joints) != len(follower_joints):
        raise JointMappingError(
            f"A leader with {len(leader_joints)} joints cannot drive a follower with {len(follower_joints)}"
        )


class TeleopControl(ControlAlgorithm):
    """Command the follower whatever the leader is currently measuring."""

    def __init__(
        self,
        leader: LeaderDevice,
        store: SessionStore,
        action_keys: Sequence[str],
        *,
        hz: float,
    ) -> None:
        check_pairing(leader.joint_names, action_keys)
        super().__init__(store, action_keys, name="teleop", hz=hz)
        self._leader = leader
        self._observation_keys = [
            joint_feature_key(OBSERVATION_PREFIX, leader.key, joint) for joint in leader.joint_names
        ]

    def compute(self) -> ControlAction:
        """Read the leader, publish what it said, and command the same.

        One serial transaction per tick, on this control's own thread, so the
        robot's loop is never waiting on it.
        """
        values, timestamp = self._leader.read()
        self._store.write_many(
            {key: float(value) for key, value in zip(self._observation_keys, values, strict=True)},
            timestamp=timestamp,
        )
        # The leader's own capture time: the gap between readings is the period
        # the robot interpolates across.
        return ControlAction(values=values, timestamp=timestamp)

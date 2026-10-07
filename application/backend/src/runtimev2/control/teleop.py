"""Drive a follower from a leader's measured position.

Reads the leader device itself rather than the store, so the position and the
command derived from it come from one read on one thread. There is nothing
between them to fall out of phase.

What it read is published too, under the leader's observation features, so the
store holds the leader's position for a client to show -- and holds the same
value the command was derived from rather than a separately-timed one.
"""

from __future__ import annotations
import time
import numpy as np
from runtimev2.dataset_layout import STATE_KEY

from typing import TYPE_CHECKING

from runtimev2.control.base import ControlAlgorithm
from runtimev2.features import ACTION_KEY, OBSERVATION_PREFIX, robot_feature_key

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
        follower_joints: Sequence[str],
        *,
        hz: float,
    ) -> None:
        check_pairing(leader.joint_names, follower_joints)
        super().__init__(store, name="teleop", hz=hz)
        self._leader = leader
        self._leader_key = robot_feature_key(OBSERVATION_PREFIX, leader.key)


        sample = self._store.read(STATE_KEY)
        if sample is None:
            raise RuntimeError("the robot has not been read yet")
        self.start_position = np.asarray(sample.value, dtype=np.float32)
        self.start_time = time.monotonic()
        self.smooth_entrance_time = 3

    def tick(self) -> None:
        """Read the leader, publish what it said, and command the same.

        One serial transaction per tick, on this control's own thread, so the
        robot's loop is never waiting on it. Both writes carry the leader's own
        capture time: the gap between readings is the period the robot
        interpolates across, and the published position is the one the command
        was derived from.
        """
        values, timestamp = self._leader.read()
        if timestamp - self.start_time < self.smooth_entrance_time:

            sample = self._store.read(STATE_KEY)
            if sample is None:
                return
            fraction = (timestamp - self.start_time) / self.smooth_entrance_time
            values = self.start_position + (values - self.start_position) * np.float32(fraction)
            self._store.write_many({self._leader_key: values, ACTION_KEY: values}, timestamp=timestamp)
        else:
            self._store.write_many({self._leader_key: values, ACTION_KEY: values}, timestamp=timestamp)

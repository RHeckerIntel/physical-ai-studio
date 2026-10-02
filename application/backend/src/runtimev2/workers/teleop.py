"""Drive one robot's action features from another robot's observations.

This is all teleoperation is under the store model: a worker that reads keys
and writes keys. It holds no device, so it can tick at whatever rate the
control loop wants regardless of what either robot is doing, and the follower's
``RobotWorker`` neither knows nor cares that a leader is what authored its
action.

The same shape covers the other sources. A keyboard mapping two base joints, a
policy writing a whole arm, or a protocol ramping to a home position differ
only in which keys they write and what they compute.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from runtimev2.store import FeatureStore


class TeleopSource:
    """Copy observation features onto action features, one pair at a time.

    Explicit key pairs rather than two robot names: a source that drives only
    part of a robot is the point, and the mapping is the only thing that makes
    one source different from another.
    """

    def __init__(self, store: FeatureStore, mapping: Mapping[str, str], *, name: str = "teleop") -> None:
        self._store = store
        self._name = name
        # Materialized so the iteration order is fixed and the keys are
        # validated against the spec once rather than on every tick.
        self._pairs = tuple(mapping.items())
        for source, target in self._pairs:
            for key in (source, target):
                if key not in store.spec:
                    raise ValueError(f"{key!r} is not a feature of this session")

    @property
    def name(self) -> str:
        return self._name

    @property
    def targets(self) -> tuple[str, ...]:
        """Action features this source authors."""
        return tuple(target for _source, target in self._pairs)

    def tick(self) -> None:
        """Forward every mapped observation to its action feature.

        A source key nobody has written yet is skipped rather than defaulted:
        the follower holds still until every joint has an author, so writing a
        placeholder would only make it start moving sooner with worse data.

        Each action carries the timestamp of the observation it came from, not
        the time of this tick, so a reader can tell how old the command it is
        acting on really is.
        """
        for source, target in self._pairs:
            sample = self._store.read(source)
            if sample is None:
                continue
            self._store.write(target, sample.value, timestamp=sample.timestamp)


def joint_mapping(leader_keys: tuple[str, ...], follower_keys: tuple[str, ...]) -> dict[str, str]:
    """Pair a leader's observation keys with a follower's action keys, in order.

    Positional, not by name: the two robots are different types as often as
    not, and what matters is that joint *i* of one drives joint *i* of the
    other. Both sequences come from their drivers in ``joint_names`` order.

    Raises:
        ValueError: The two robots do not have the same number of joints,
            which leaves no defensible pairing.
    """
    if len(leader_keys) != len(follower_keys):
        raise ValueError(
            f"cannot teleoperate {len(follower_keys)} joints from {len(leader_keys)}: "
            "the leader and follower must have the same number of joints"
        )
    return dict(zip(leader_keys, follower_keys, strict=True))

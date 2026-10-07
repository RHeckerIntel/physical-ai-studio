"""The shape of a session's data, and what that shape is compatible with.

An environment decides which robots and cameras a session runs, and therefore
which features exist. That set is what the store is keyed by and what every
worker addresses.

A feature is a *vector with named components*, not a scalar per joint: one
feature holds a whole robot's positions, in the order its driver reports them.
That is what every boundary already speaks -- a driver returns and accepts an
ndarray ordered by ``joint_names``, a dataset stores ``action`` and
``observation.state`` packed with a ``names`` list, a model takes a state
vector. Splitting them into named scalars meant packing and unpacking at every
one of those boundaries and reconstructing the order from key sorting each
time, which is a trap rather than a design.

Names live here rather than on a sample: a robot's joints do not change
mid-session, and strings cannot live in the shared array the values do.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

# ``observation`` is what a device reports, ``action`` is what it is told to do.
# Both exist for the same joint, which is why the kind is part of the key: a
# session holds the commanded position and the measured one at the same time.
FeatureKind = Literal["observation", "action"]

_IMAGE_DIMS = 3
"""``(h, w, c)``. A frame is the one feature with no component names."""

OBSERVATION_PREFIX = "observation"
ACTION_PREFIX = "action"
IMAGE_INFIX = "images"

STATE_KEY = f"{OBSERVATION_PREFIX}.state"
"""The driven robot's measured joints. Always present."""

ACTION_KEY = ACTION_PREFIX
"""The driven robot's commanded joints. Always present.

These two names are LeRobot's, deliberately: a session carries the same keys a
dataset records, so nothing has to translate between a store and a recording.
Both are singular because a session drives one robot -- an assumption recording
and inference already made, now stated once.
"""


@dataclass(frozen=True, slots=True)
class Feature:
    """One vector a session carries, and what its components are.

    Attributes:
        key: Dotted name, unique within a spec.
        kind: Whether this is reported by a device or commanded to one.
        names: What each component is, in order -- a robot's joint names. The
            order is the driver's, so it is also the order ``send_action``
            expects and the order a dataset records.
        shape: The value's array shape: ``(len(names),)`` for a vector,
            ``(h, w, c)`` for an image.
        dtype: Numpy dtype name.
    """

    key: str
    kind: FeatureKind
    names: tuple[str, ...] = ()
    shape: tuple[int, ...] = ()
    dtype: str = "float32"

    @property
    def is_image(self) -> bool:
        """Whether this is a frame rather than a named vector."""
        return not self.names and len(self.shape) == _IMAGE_DIMS

    @property
    def size(self) -> int:
        """How many numbers the value holds."""
        return int(np.prod(self.shape)) if self.shape else 0


def robot_feature_key(kind: FeatureKind, robot: str) -> str:
    """Return the key for a *read-only* robot's position vector.

    The driven robot uses :data:`STATE_KEY` and :data:`ACTION_KEY`; this names
    the others, a leader being the only one so far.
    """
    return f"{kind}.{robot}"


def sanitize_name(name: str) -> str:
    """Return a stable, filesystem- and key-safe name for a device.

    A display name is free-form; a feature key has to survive being a dict key,
    a zenoh key expression and a dataset column.
    """
    return re.sub(r"[^a-z0-9_-]+", "_", name.strip().lower())


def image_feature_key(camera: str) -> str:
    """Return the key for one camera's frames."""
    return f"{OBSERVATION_PREFIX}.{IMAGE_INFIX}.{camera}"


def sanitize_dataset_camera_name(name: str) -> str:
    """The camera key a dataset is written with.

    Deliberately not :func:`sanitize_name`: this keeps spaces, because datasets
    already on disk were recorded with them. Tightening it would orphan every
    recording from a camera whose name has a space in it.
    """
    return re.sub(r"[^a-z0-9 _-]+", "_", name.lower())


def robot_features(joint_names: Sequence[str]) -> list[Feature]:
    """Return the state and action vectors for the robot we drive.

    One of each, not one per joint: a driver reports and accepts the whole
    vector. Both exist whether or not anything is driving, because that is the
    worker's business and the shape should not change with it.

    Unkeyed by robot, because a session drives one: a control can assume both
    keys exist rather than being told which robot it is acting on.
    """
    names = tuple(joint_names)
    shape = (len(names),)
    return [
        Feature(STATE_KEY, "observation", names=names, shape=shape),
        Feature(ACTION_KEY, "action", names=names, shape=shape),
    ]


def leader_features(robot: str, joint_names: Sequence[str]) -> list[Feature]:
    """Return the observation vector for a robot we only read.

    No action feature. A leader is an input device -- nothing commands it --
    so an action slot would be a key that can never be written.
    """
    names = tuple(joint_names)
    return [Feature(robot_feature_key(OBSERVATION_PREFIX, robot), "observation", names=names, shape=(len(names),))]


def camera_features(cameras: dict[str, tuple[int, int, int]]) -> list[Feature]:
    """Return the observation features for cameras, keyed by name to ``(h, w, c)``.

    Shapes here are what the environment declares. A publisher already serving
    the device at another resolution wins at connect time, so a recording
    stores what the session actually saw -- see ``build_lerobot_dataset_features``.
    """
    return [
        Feature(image_feature_key(name), "observation", shape=shape, dtype="uint8")
        for name, shape in sorted(cameras.items())
    ]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """The complete set of features a session carries."""

    features: tuple[Feature, ...]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for feature in self.features:
            if feature.key in seen:
                raise ValueError(f"duplicate feature key {feature.key!r}")
            seen.add(feature.key)

    @classmethod
    def build(cls, *groups: Iterable[Feature]) -> FeatureSpec:
        """Build a spec from feature groups, ordered by key so it is reproducible."""
        merged = [feature for group in groups for feature in group]
        return cls(tuple(sorted(merged, key=lambda feature: feature.key)))

    def __contains__(self, key: object) -> bool:
        return any(feature.key == key for feature in self.features)

    def __getitem__(self, key: str) -> Feature:
        for feature in self.features:
            if feature.key == key:
                return feature
        raise KeyError(key)

    def keys(self, kind: FeatureKind | None = None) -> tuple[str, ...]:
        """Return the feature keys, optionally narrowed to one kind."""
        return tuple(feature.key for feature in self.features if kind is None or feature.kind == kind)

    def of(self, kind: FeatureKind) -> tuple[Feature, ...]:
        return tuple(feature for feature in self.features if feature.kind == kind)

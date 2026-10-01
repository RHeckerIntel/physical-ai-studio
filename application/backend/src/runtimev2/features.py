"""The shape of a session's data, and what that shape is compatible with.

An environment decides which robots and cameras a session runs, and therefore
which features exist. That set is the session's contract: a dataset can only be
appended to if its features match, and a model can only be loaded if the
observations it was trained on are present and the actions it emits are
writable. Everything downstream -- the store's keys, a recorded row's columns,
a model's input -- is derived from here rather than rediscovered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

# ``observation`` is what a device reports, ``action`` is what it is told to do.
# Both exist for the same joint, which is why the kind is part of the key: a
# session holds the commanded position and the measured one at the same time.
FeatureKind = Literal["observation", "action"]

OBSERVATION_PREFIX = "observation"
ACTION_PREFIX = "action"
IMAGE_INFIX = "images"


@dataclass(frozen=True, slots=True)
class Feature:
    """One named value a session carries.

    Attributes:
        key: Dotted name, unique within a spec. Also what a dataset column and
            a model input are matched on, so it must be stable across sessions
            for the same environment.
        kind: Whether this is reported by a device or commanded to one.
        shape: ``()`` for a scalar, ``(h, w, c)`` for an image.
        dtype: Numpy dtype name.
    """

    key: str
    kind: FeatureKind
    shape: tuple[int, ...] = ()
    dtype: str = "float32"

    @property
    def is_image(self) -> bool:
        return bool(self.shape)


def joint_feature_key(kind: FeatureKind, robot: str, joint: str) -> str:
    """Return the key for one robot joint's position."""
    return f"{kind}.{robot}.{joint}.pos"


def image_feature_key(camera: str) -> str:
    """Return the key for one camera's frames."""
    return f"{OBSERVATION_PREFIX}.{IMAGE_INFIX}.{camera}"


def robot_features(robot: str, joint_names: Sequence[str]) -> list[Feature]:
    """Return the observation and action features for one robot.

    Both kinds are emitted for every joint. A robot that is only ever read
    still gets action features: whether they are written to hardware is the
    worker's business, not the shape's, and leaving them out would mean the
    shape changed when a robot started being driven.
    """
    features: list[Feature] = []
    for joint in joint_names:
        features.append(Feature(joint_feature_key(OBSERVATION_PREFIX, robot, joint), "observation"))
        features.append(Feature(joint_feature_key(ACTION_PREFIX, robot, joint), "action"))
    return features


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

    def missing_from(self, required: FeatureSpec) -> list[str]:
        """Describe every way ``required`` is not satisfied by this spec.

        Reasons rather than a bool: "this model does not fit this environment"
        is not actionable, and the person reading it needs to know whether to
        reselect a camera, pick another model, or recalibrate a robot.
        """
        reasons: list[str] = []
        for feature in required.features:
            if feature.key not in self:
                reasons.append(f"{feature.key} is missing")
                continue
            mine = self[feature.key]
            if mine.kind != feature.kind:
                reasons.append(f"{feature.key} is an {mine.kind}, expected an {feature.kind}")
            if mine.shape != feature.shape:
                reasons.append(f"{feature.key} has shape {mine.shape}, expected {feature.shape}")
            if mine.dtype != feature.dtype:
                reasons.append(f"{feature.key} has dtype {mine.dtype}, expected {feature.dtype}")
        return reasons

    def satisfies(self, required: FeatureSpec) -> bool:
        """Whether everything ``required`` needs is present here, with the same shapes.

        Extra features are fine: an environment with a second camera can still
        run a model trained without it.
        """
        return not self.missing_from(required)

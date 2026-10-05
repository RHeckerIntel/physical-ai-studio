"""Project a session's features onto the layout a dataset and a model use.

Two namings exist on purpose, at different levels.

Inside a session every joint is addressed on its own -- that is what lets a
keyboard author a base while a leader authors an arm, with neither knowing
about the other. On disk LeRobot packs those joints into ``action`` and
``observation.state`` vectors whose components are *named but unprefixed*::

    action                      shape (6,)  ['shoulder_pan.pos', ..., 'gripper.pos']
    observation.state           shape (6,)  ['shoulder_pan.pos', ...]
    observation.images.overhead shape (480, 640, 3)

A model was trained against that layout, and a dataset was recorded in it. So
"does this model fit this environment" and "can this dataset be appended to"
are questions about the packed layout, not about session keys -- comparing
session keys would compare names that never appear in either artifact.

The upside of the packed names being unprefixed is that a robot's session key
never reaches disk, so renaming a robot cannot invalidate a dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from runtime.features import sanitize_camera_name
from runtimev2.features import IMAGE_INFIX, OBSERVATION_PREFIX

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from runtimev2.environment import SessionShape

ACTION_KEY = "action"
STATE_KEY = f"{OBSERVATION_PREFIX}.state"
IMAGE_PREFIX = f"{OBSERVATION_PREFIX}.{IMAGE_INFIX}."


@dataclass(frozen=True, slots=True)
class LayoutEntry:
    """One top-level dataset feature.

    Attributes:
        shape: As a dataset's ``info.json`` records it.
        names: Component names for a packed vector, ``None`` for an image.
    """

    shape: tuple[int, ...]
    names: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class DatasetLayout:
    """The packed feature layout of a dataset or a model's training data."""

    entries: dict[str, LayoutEntry]

    @classmethod
    def from_shape(cls, shape: SessionShape) -> DatasetLayout:
        """Project an environment onto the layout a recording of it would have.

        Only the follower's joints are packed. An environment can hold a leader
        too, but a leader is a control source rather than something recorded,
        and today exactly one follower is expected.

        An environment with more than one follower has no answer here: nothing
        in the environment says whose joints come first in ``action``, and
        guessing would produce datasets that are silently incompatible with
        each other. That mapping is the environment rework's to define.

        Raises:
            ValueError: The environment has no follower, or more than one.
        """
        followers = shape.robots
        if len(followers) != 1:
            raise ValueError(
                f"a recording needs exactly one follower, found {len(followers)}; "
                "the environment does not say how several would be ordered"
            )
        joints = tuple(f"{joint}.pos" for joint in followers[0].joint_names)
        entries: dict[str, LayoutEntry] = {
            ACTION_KEY: LayoutEntry(shape=(len(joints),), names=joints),
            STATE_KEY: LayoutEntry(shape=(len(joints),), names=joints),
        }
        for camera in shape.cameras:
            # Deliberately not the session key: datasets on disk were written
            # with ``sanitize_camera_name``, which keeps spaces, so a camera
            # called "Overhead Cam" is already stored as ``overhead cam``.
            # Tightening it here would orphan every recording of that camera.
            entries[f"{IMAGE_PREFIX}{sanitize_camera_name(camera.name)}"] = LayoutEntry(shape=camera.shape)
        return cls(entries)

    @classmethod
    def from_info(cls, features: Mapping[str, Any]) -> DatasetLayout:
        """Read the layout out of a dataset's ``meta/info.json`` features block.

        Bookkeeping columns every LeRobot dataset carries -- ``timestamp``,
        ``index`` and friends -- are dropped: they say nothing about whether an
        environment can produce this dataset.

        An image's ``names`` are dropped too. They are always
        ``["height", "width", "channels"]``, describing the axes rather than
        naming components to match, so comparing them would reject every
        environment over metadata.
        """
        entries: dict[str, LayoutEntry] = {}
        for key, value in features.items():
            is_image = key.startswith(IMAGE_PREFIX)
            if key not in {ACTION_KEY, STATE_KEY} and not is_image:
                continue
            names = None if is_image else value.get("names")
            entries[key] = LayoutEntry(
                shape=tuple(int(dim) for dim in value.get("shape", ())),
                names=tuple(str(name) for name in names) if isinstance(names, list) else None,
            )
        return cls(entries)

    def incompatibilities(self, required: DatasetLayout) -> list[str]:
        """Describe every way this layout cannot stand in for ``required``.

        Reasons rather than a bool: "incompatible" leaves the user guessing
        whether to reselect a camera, pick another model, or recalibrate.
        """
        reasons: list[str] = []
        for key, wanted in required.entries.items():
            mine = self.entries.get(key)
            if mine is None:
                reasons.append(f"{key} is missing")
                continue
            if mine.shape != wanted.shape:
                reasons.append(f"{key} has shape {mine.shape}, expected {wanted.shape}")
            if wanted.names is not None and mine.names != wanted.names:
                reasons.append(f"{key} has components {_describe(mine.names)}, expected {_describe(wanted.names)}")
        return reasons

    def satisfies(self, required: DatasetLayout) -> bool:
        """Whether everything ``required`` needs is present, with matching shapes.

        Extra features are allowed: an environment with a spare camera can
        still run a model trained without it.
        """
        return not self.incompatibilities(required)


def _describe(names: Sequence[str] | None) -> str:
    return "none" if names is None else "(" + ", ".join(names) + ")"

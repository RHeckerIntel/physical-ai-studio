"""Open a dataset for recording, and check it against the session's shape.

A dataset sits above any particular set of connected devices: it is chosen once
and survives an environment being unloaded and loaded again. What it has to
agree with is the :class:`SessionShape` -- the joints and cameras that decide
what a row looks like -- so that is what it is checked against here rather than
at the first frame, when an episode is already half written.

Rates are the dataset's, not the session's. A LeRobot dataset is single-rate, so
appending to one has to use the fps it was created with; only a new dataset can
take the rate the caller asked for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from internal_datasets.access_mode import DatasetAccessMode
from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset
from runtime.dataset_features import build_lerobot_dataset_features
from runtime.features import sanitize_camera_name
from runtimev2.dataset_layout import DatasetLayout

if TYPE_CHECKING:
    from pathlib import Path
    from uuid import UUID

    from internal_datasets.mutations.recording_mutation import RecordingMutation
    from runtimev2.environment import SessionShape


class DatasetIncompatibleError(RuntimeError):
    """Raised when a dataset's rows cannot be produced by this session.

    Carries every reason rather than the first, because the fix differs: a
    camera may need reselecting, a robot recalibrating, or the dataset may
    simply belong to another rig.
    """

    def __init__(self, dataset_id: UUID, reasons: list[str]) -> None:
        self.reasons = reasons
        super().__init__(f"Dataset {dataset_id} does not match this environment: " + "; ".join(reasons))


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    """A dataset open for recording, and the rate its frames are written at."""

    dataset_id: UUID
    path: Path
    fps: int
    """What the worker ticks at. The dataset's own rate when appending."""


def _existing_info(path: Path) -> dict | None:
    """Return a dataset's ``meta/info.json``, or ``None`` if it is new."""
    info_path = path / "meta" / "info.json"
    if not info_path.exists():
        return None
    return json.loads(info_path.read_text())


def dataset_fps(path: Path, requested: int) -> int:
    """Return the rate frames must be written at.

    An existing dataset's own fps wins: LeRobot stores one rate per dataset, so
    appending at another would leave timestamps that disagree with the frames
    between them.
    """
    info = _existing_info(path)
    if info is None:
        return requested
    recorded = int(info.get("fps", requested))
    if recorded != requested:
        logger.info(
            "Dataset at {} records at {}Hz; ignoring the requested {}Hz so its timestamps stay consistent",
            path,
            recorded,
            requested,
        )
    return recorded


def check_compatible(dataset_id: UUID, path: Path, shape: SessionShape) -> None:
    """Fail if this session cannot produce the dataset's rows.

    A new dataset is always compatible -- it has no rows yet, and its features
    are written from this shape.

    Raises:
        DatasetIncompatibleError: The existing dataset's layout does not match.
    """
    info = _existing_info(path)
    if info is None:
        return
    recorded = DatasetLayout.from_info(info.get("features", {}))
    reasons = DatasetLayout.from_shape(shape).incompatibilities(recorded)
    if reasons:
        raise DatasetIncompatibleError(dataset_id, reasons)


def open_for_recording(
    dataset_id: UUID,
    path: Path,
    shape: SessionShape,
    *,
    requested_fps: int,
    camera_specs: dict[str, tuple[int, int, int]] | None = None,
) -> tuple[LoadedDataset, RecordingMutation]:
    """Open ``path`` for appending and return its mutation and metadata.

    Args:
        dataset_id: For error messages.
        path: The dataset directory.
        shape: What this session can produce.
        requested_fps: Used only when the dataset is new.
        camera_specs: Actual ``(h, w, c)`` per dataset camera key, once frames
            have arrived. Falls back to what the environment declared, which is
            what the cameras are resized to anyway.

    Returns:
        A ``(LoadedDataset, RecordingMutation)`` pair.

    Raises:
        DatasetIncompatibleError: The dataset's rows do not match this session.
        ValueError: The session has no single follower to record.
    """
    check_compatible(dataset_id, path, shape)
    followers = shape.robots
    if len(followers) != 1:
        raise ValueError(f"recording needs exactly one follower, found {len(followers)}")
    follower = followers[0]

    fps = dataset_fps(path, requested_fps)
    specs = camera_specs or {sanitize_camera_name(camera.name): camera.shape for camera in shape.cameras}
    dataset = InternalLeRobotDataset(path, access_mode=DatasetAccessMode.RECORDING_MUTATION)
    mutation = dataset.start_recording_mutation(
        fps=fps,
        features=build_lerobot_dataset_features(
            joint_names=list(follower.joint_names),
            camera_specs=specs,
        ),
        robot_type=follower.key,
    )
    logger.info("Dataset {} open for recording at {}Hz", dataset_id, fps)
    return LoadedDataset(dataset_id=dataset_id, path=path, fps=fps), mutation

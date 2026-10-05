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
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from internal_datasets.access_mode import DatasetAccessMode
from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset
from runtimev2.dataset_layout import DatasetLayout, build_lerobot_dataset_features
from runtimev2.features import sanitize_dataset_camera_name

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
    specs = camera_specs or {sanitize_dataset_camera_name(camera.name): camera.shape for camera in shape.cameras}
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


class RecordingState:
    """Recording flags and the open mutation, shared across threads.

    Owned by the session rather than a loaded environment, so an episode
    opened before an environment swap is the same episode after it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._dataset_loaded = False
        self._is_recording = False
        self._episodes_recorded = 0
        self._task: str | None = None
        self._mutation: RecordingMutation | None = None
        self._closed = False

    @property
    def dataset_loaded(self) -> bool:
        with self._lock:
            return self._dataset_loaded

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._is_recording

    @property
    def episodes_recorded(self) -> int:
        with self._lock:
            return self._episodes_recorded

    def start(self, task: str) -> bool:
        """Begin an episode. Return False when no dataset is loaded."""
        with self._lock:
            if self._mutation is None or self._closed:
                return False
            self._task = task
            self._is_recording = True
            return True

    def attach_mutation(self, mutation: RecordingMutation) -> None:
        with self._lock:
            self._mutation = mutation
            self._dataset_loaded = True
            self._is_recording = False

    def mark_saved(self) -> None:
        with self._lock:
            self._is_recording = False
            self._episodes_recorded += 1

    def mark_discarded(self) -> None:
        with self._lock:
            self._is_recording = False

    def add_frame(self, observation: dict[str, Any], action: dict[str, float]) -> None:
        """Write one tick under the state lock so save/discard cannot interleave."""
        with self._lock:
            if self._closed or not self._is_recording or self._mutation is None or self._task is None:
                return
            self._mutation.add_frame(observation, action, self._task)

    def stop_episode(self) -> RecordingMutation:
        """Clear the recording flag so ticks skip, then return the mutation.

        Save and discard run off the control thread. Stopping first means an
        in-flight ``add_frame`` finishes (it holds this lock), then later ticks
        see ``is_recording`` is false and skip, then video encode can run.
        """
        with self._lock:
            if not self._is_recording or self._mutation is None:
                raise RuntimeError("No episode is being recorded.")
            self._is_recording = False
            return self._mutation

    def current_mutation(self) -> RecordingMutation | None:
        """Return the attached mutation without requiring an open episode.

        Discard doubles as the recovery path after a failed save, which has
        already cleared the recording flag. Requiring an open episode there
        would leave the buffer with no way to clear it.
        """
        with self._lock:
            self._is_recording = False
            return self._mutation

    def take_mutation(self) -> RecordingMutation | None:
        """Detach the mutation so teardown can finalize it once.

        ``_episodes_recorded`` counts episodes saved into the attached mutation
        and not yet copied into the dataset. Detaching is the moment that count
        becomes zero: the UI adds it to the episodes the dataset API returns, so
        leaving it set double-counts every episode once the copy lands.
        """
        with self._lock:
            mutation = self._mutation
            self._mutation = None
            self._is_recording = False
            self._dataset_loaded = False
            self._task = None
            self._episodes_recorded = 0
            return mutation

    def close(self) -> None:
        with self._lock:
            self._closed = True

"""Write the store's current values to a dataset, at the dataset's own rate.

The worker reads; it never asks a device for anything. That is what lets it
record at a rate unrelated to the hardware: a 50Hz recording of a 100Hz arm and
30Hz cameras takes whatever each feature's latest value is at its tick, and the
timestamps already in the store say how fresh each one was.

It also means recording cannot stall a robot. A slow disk makes the recording
fall behind -- reported by the rate loop -- rather than slowing the arm down.

Rows use the dataset's naming, not the session's: joints are bare
``<joint>.pos`` and cameras are their sanitized display names, because that is
what is already on disk. ``dataset_layout`` has the reasoning.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from loguru import logger

from runtime.features import sanitize_camera_name
from runtimev2.features import ACTION_PREFIX, OBSERVATION_PREFIX, image_feature_key, joint_feature_key
from runtimev2.workers.base import ThreadedWorker

if TYPE_CHECKING:
    from collections.abc import Generator

    from runtime.callbacks.recording import RecordingState
    from runtimev2.environment import SessionShape
    from runtimev2.session_store import SessionStore


class DatasetWorker(ThreadedWorker):
    """Sample the store once per tick and append a frame while recording.

    Holds no device, so ``acquire`` has nothing to do: the dataset is opened by
    whoever chose it, above the environment, and outlives this worker.
    """

    def __init__(
        self,
        store: SessionStore,
        recording: RecordingState,
        shape: SessionShape,
        *,
        hz: float,
        name: str = "dataset",
    ) -> None:
        super().__init__(name=name, hz=hz)
        self._store = store
        self._recording = recording
        follower = shape.robots[0]
        # Resolved once: store key -> dataset key, for both halves of a row.
        self._observations = {
            joint_feature_key(OBSERVATION_PREFIX, follower.key, joint): f"{joint}.pos" for joint in follower.joint_names
        }
        self._actions = {
            joint_feature_key(ACTION_PREFIX, follower.key, joint): f"{joint}.pos" for joint in follower.joint_names
        }
        self._images = {image_feature_key(camera.key): sanitize_camera_name(camera.name) for camera in shape.cameras}
        self._skipped = 0

    @contextmanager
    def acquire(self) -> Generator[None]:
        """Nothing to hold: the dataset belongs to the session, not to this worker."""
        try:
            yield
        finally:
            if self._skipped:
                logger.info("Recording skipped {} ticks for want of a complete row", self._skipped)

    def tick(self) -> None:
        """Append one frame, if recording and every feature has a value.

        A row is all or nothing. Writing one with a joint missing would put a
        default where a measurement belongs, and nothing downstream could tell
        the difference -- so an incomplete tick is skipped and counted instead.
        """
        if not self._recording.is_recording:
            return
        wanted = (*self._observations, *self._actions, *self._images)
        snapshot = self._store.snapshot(wanted)
        if len(snapshot) != len(wanted):
            self._skipped += 1
            return

        observation: dict[str, Any] = {
            dataset_key: snapshot[store_key].value for store_key, dataset_key in self._observations.items()
        }
        for store_key, dataset_key in self._images.items():
            # Copied: the store keeps handing out the same array until the next
            # frame, and the writer must not see it change underneath.
            observation[dataset_key] = snapshot[store_key].value.copy()
        action = {dataset_key: snapshot[store_key].value for store_key, dataset_key in self._actions.items()}
        self._recording.add_frame(observation, action)

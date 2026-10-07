"""Recording reads the store at its own rate and never asks a device anything."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from runtimev2.environment import CameraShape, RobotShape, SessionShape
from runtimev2.features import ACTION_KEY, STATE_KEY, image_feature_key
from runtimev2.store import FeatureStore
from runtimev2.workers.dataset import DatasetWorker

JOINTS = ("shoulder_pan", "gripper")


class _Recording:
    """Stands in for RecordingState: records what frames it was handed."""

    def __init__(self, *, recording: bool = True) -> None:
        self.is_recording = recording
        self.frames: list[tuple[dict, dict]] = []

    def add_frame(self, observation: dict, action: dict) -> None:
        self.frames.append((observation, action))


def _shape(*, cameras: bool = True) -> SessionShape:
    return SessionShape(
        robots=(RobotShape(key="follower", robot_id="r0", role="follower", joint_names=JOINTS),),
        cameras=(
            (CameraShape(key="overhead", name="Overhead Cam", camera_id="c0", shape=(480, 640, 3), fps=30.0),)
            if cameras
            else ()
        ),
    )


def _store(shape: SessionShape) -> FeatureStore:
    return FeatureStore(shape.feature_spec())


def _fill(store: FeatureStore, shape: SessionShape, *, value: float = 1.0) -> None:
    store.write(STATE_KEY, [value] * len(JOINTS), timestamp=1.0)
    store.write(ACTION_KEY, [value + 10] * len(JOINTS), timestamp=1.0)
    for camera in shape.cameras:
        store.write(
            image_feature_key(camera.key),
            np.zeros(camera.shape, dtype=np.uint8),
            timestamp=1.0,
        )


def _worker(store: FeatureStore, recording: _Recording, shape: SessionShape, hz: float = 30.0) -> DatasetWorker:
    return DatasetWorker(store, recording, shape, hz=hz)  # type: ignore[arg-type]


class TestRows:
    def test_a_row_uses_the_datasets_naming_not_the_sessions(self) -> None:
        """The robot prefix never reaches disk, and camera keys use the sanitizer
        that existing datasets were written with."""
        shape = _shape()
        store = _store(shape)
        recording = _Recording()
        _fill(store, shape)

        _worker(store, recording, shape).tick()

        observation, action = recording.frames[0]
        assert set(action) == {"shoulder_pan.pos", "gripper.pos"}
        assert {"shoulder_pan.pos", "gripper.pos", "overhead cam"} <= set(observation)

    def test_observations_and_actions_are_separated(self) -> None:
        shape = _shape(cameras=False)
        store = _store(shape)
        recording = _Recording()
        _fill(store, shape, value=2.0)

        _worker(store, recording, shape).tick()

        observation, action = recording.frames[0]
        assert observation["gripper.pos"] == 2.0
        assert action["gripper.pos"] == 12.0

    def test_a_frame_is_copied_out_of_the_store(self) -> None:
        """The store hands out the same array until the next frame; the writer
        must not see it change underneath."""
        shape = _shape()
        store = _store(shape)
        recording = _Recording()
        _fill(store, shape)

        _worker(store, recording, shape).tick()
        stored = store.read(image_feature_key("overhead")).value
        stored[0, 0, 0] = 99

        assert recording.frames[0][0]["overhead cam"][0, 0, 0] == 0


class TestWhenNotToWrite:
    def test_nothing_is_written_while_not_recording(self) -> None:
        shape = _shape()
        store = _store(shape)
        recording = _Recording(recording=False)
        _fill(store, shape)

        _worker(store, recording, shape).tick()

        assert recording.frames == []

    def test_an_incomplete_row_is_skipped(self) -> None:
        """Writing a default where a measurement belongs would be indistinguishable
        from a real reading afterwards."""
        shape = _shape()
        store = _store(shape)
        recording = _Recording()
        store.write(STATE_KEY, [1.0] * len(JOINTS), timestamp=1.0)

        _worker(store, recording, shape).tick()

        assert recording.frames == [], "no action written, so no row"

    def test_a_missing_camera_frame_also_skips(self) -> None:
        shape = _shape()
        store = _store(shape)
        recording = _Recording()
        store.write(STATE_KEY, [1.0] * len(JOINTS), timestamp=1.0)
        store.write(ACTION_KEY, [1.0] * len(JOINTS), timestamp=1.0)

        _worker(store, recording, shape).tick()

        assert recording.frames == []


class TestRate:
    async def test_it_records_at_its_own_rate(self) -> None:
        """A 50Hz recording of 100Hz arms is the point: the worker samples
        whatever is latest rather than being driven by a device."""
        shape = _shape(cameras=False)
        store = _store(shape)
        recording = _Recording()
        _fill(store, shape)
        worker = _worker(store, recording, shape, hz=200.0)

        async with worker:
            await asyncio.sleep(0.15)

        assert len(recording.frames) > 5, "the worker's own thread never recorded"

    def test_the_rate_is_whatever_it_was_given(self) -> None:
        shape = _shape(cameras=False)
        worker = _worker(_store(shape), _Recording(), shape, hz=12.0)

        assert worker.hz == 12.0
        assert worker.name == "dataset"


def test_a_shape_with_no_follower_cannot_be_recorded() -> None:
    shape = SessionShape(robots=(), cameras=())

    with pytest.raises(IndexError):
        _worker(_store(shape), _Recording(), shape)

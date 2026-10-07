"""A camera worker is the only thing that knows frames come over iceoryx2."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np
import pytest

from runtimev2.features import STATE_KEY, FeatureSpec, camera_features, image_feature_key
from runtimev2.store import FeatureStore
from runtimev2.workers.camera import CameraWorker, connect_with_retry

KEY = image_feature_key("overhead")


@dataclass
class _Frame:
    data: np.ndarray
    timestamp: float
    sequence: int = 0


@dataclass
class _FakeCamera:
    """Stands in for a connected SharedCamera."""

    frames: list[_Frame] = field(default_factory=list)
    reads: int = 0
    fail_with: Exception | None = None
    connect_error: Exception | None = None
    connected: bool = False

    disconnects: int = 0

    def connect(self) -> None:
        if self.connect_error is not None:
            raise self.connect_error
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False
        self.disconnects += 1

    def read_latest(self) -> _Frame:
        self.reads += 1
        if self.fail_with is not None:
            raise self.fail_with
        # Latest, not next: a worker ticking faster re-reads the same frame.
        return self.frames[min(self.reads - 1, len(self.frames) - 1)]


def _setup(frames: list[_Frame] | None = None) -> tuple[_FakeCamera, FeatureStore, CameraWorker]:
    camera = _FakeCamera(
        frames=frames or [_Frame(np.zeros((480, 640, 3), dtype=np.uint8), timestamp=100.0)],
    )
    store = FeatureStore(FeatureSpec.build(camera_features({"overhead": (480, 640, 3)})))
    worker = CameraWorker(lambda: camera, store, key=KEY, hz=30.0)
    # ``tick`` needs the camera attached; these tests drive ticks directly
    # rather than through the worker's own thread.
    camera.connect()
    worker._camera = camera
    return camera, store, worker


class TestPublishing:
    def test_a_tick_writes_the_frame(self) -> None:
        _camera, store, worker = _setup()

        worker.tick()

        sample = store.read(KEY)
        assert sample is not None
        assert sample.value.shape == (480, 640, 3)

    def test_the_cameras_own_capture_time_is_kept(self) -> None:
        """Storing arrival time would fold transport delay into the reading."""
        _camera, store, worker = _setup([_Frame(np.zeros((480, 640, 3), np.uint8), timestamp=1234.5)])

        worker.tick()

        assert store.read(KEY).timestamp == 1234.5

    def test_a_later_frame_replaces_the_earlier_one(self) -> None:
        frames = [
            _Frame(np.full((480, 640, 3), 1, np.uint8), timestamp=1.0),
            _Frame(np.full((480, 640, 3), 2, np.uint8), timestamp=2.0),
        ]
        _camera, store, worker = _setup(frames)

        worker.tick()
        worker.tick()

        sample = store.read(KEY)
        assert sample.timestamp == 2.0
        assert int(sample.value[0, 0, 0]) == 2

    def test_ticking_faster_than_the_camera_republishes_the_same_frame(self) -> None:
        """The repeated timestamp is what tells a reader nothing new arrived."""
        _camera, store, worker = _setup([_Frame(np.zeros((480, 640, 3), np.uint8), timestamp=7.0)])

        worker.tick()
        worker.tick()

        assert store.read(KEY).timestamp == 7.0

    def test_it_has_a_name_for_the_rate_loop(self) -> None:
        _camera, _store, worker = _setup()

        assert worker.name == KEY
        assert worker.key == KEY


class TestFailures:
    def test_a_dead_publisher_surfaces_rather_than_being_swallowed(self) -> None:
        """The worker is the one place that can see this; hiding it would leave
        the store serving a frame that never updates again."""
        camera, _store, worker = _setup()
        camera.fail_with = RuntimeError("shared camera is not connected")

        with pytest.raises(RuntimeError, match="not connected"):
            worker.tick()

    def test_a_key_outside_the_spec_is_refused(self) -> None:
        camera, store, _worker = _setup()

        with pytest.raises(KeyError):
            CameraWorker(lambda: camera, store, key=image_feature_key("nonexistent"), hz=30.0)


class TestDeclaredResolutionLeads:
    """The environment's resolution is the contract; the publisher is adapted to it.

    Refusing a mismatch instead would make an IP camera unusable, since its
    stream size is not ours to choose.
    """

    def test_a_larger_frame_is_resized_down(self) -> None:
        _camera, store, worker = _setup([_Frame(np.zeros((720, 1280, 3), np.uint8), timestamp=1.0)])

        worker.tick()

        assert store.read(KEY).value.shape == (480, 640, 3)

    def test_a_smaller_frame_is_resized_up(self) -> None:
        _camera, store, worker = _setup([_Frame(np.zeros((240, 320, 3), np.uint8), timestamp=1.0)])

        worker.tick()

        assert store.read(KEY).value.shape == (480, 640, 3)

    def test_a_grayscale_publisher_is_promoted_to_colour(self) -> None:
        """A depth or mono stream fills a colour feature coherently rather than
        with a differently shaped array."""
        _camera, store, worker = _setup([_Frame(np.full((480, 640), 7, np.uint8), timestamp=1.0)])

        worker.tick()

        value = store.read(KEY).value
        assert value.shape == (480, 640, 3)
        assert int(value[0, 0, 0]) == 7

    def test_a_matching_frame_is_not_copied_through_a_resize(self) -> None:
        """The normal case must not pay for the exception."""
        original = np.zeros((480, 640, 3), np.uint8)
        _camera, store, worker = _setup([_Frame(original, timestamp=1.0)])

        worker.tick()

        assert store.read(KEY).value is original

    def test_an_unconformable_frame_is_refused(self) -> None:
        _camera, _store, worker = _setup([_Frame(np.zeros((480, 640, 4), np.uint8), timestamp=1.0)])

        with pytest.raises(ValueError, match="cannot conform"):
            worker.tick()

    def test_a_non_image_feature_is_refused(self) -> None:
        from runtimev2.features import FeatureSpec, robot_features

        store = FeatureStore(FeatureSpec.build(robot_features(["gripper"])))

        with pytest.raises(ValueError, match="not an image feature"):
            CameraWorker(
                _FakeCamera,
                store,
                key=STATE_KEY,
                hz=30.0,
            )


class TestConnectWithRetry:
    """A device just taken back from another publisher needs a moment to stream."""

    def test_a_first_time_success_attaches_once(self) -> None:
        builds = 0

        def build() -> _FakeCamera:
            nonlocal builds
            builds += 1
            return _FakeCamera()

        camera = connect_with_retry(build, name="overhead", backoff=0)

        assert builds == 1
        assert isinstance(camera, _FakeCamera)

    def test_it_retries_until_the_device_wakes_up(self) -> None:
        attempts = 0

        def build() -> _FakeCamera:
            nonlocal attempts
            attempts += 1
            camera = _FakeCamera()
            if attempts < 3:
                camera.connect_error = RuntimeError("Timed out waiting for first frame after 5.0s")
            return camera

        camera = connect_with_retry(build, name="overhead", attempts=3, backoff=0)

        assert attempts == 3
        assert camera.connected

    def test_the_last_failure_is_what_surfaces(self) -> None:
        """An earlier failure is a device that had not woken up; the final one is the answer."""

        def build() -> _FakeCamera:
            camera = _FakeCamera()
            camera.connect_error = RuntimeError("device or resource busy")
            return camera

        with pytest.raises(RuntimeError, match="busy"):
            connect_with_retry(build, name="overhead", attempts=2, backoff=0)

    def test_a_fresh_camera_is_built_per_attempt(self) -> None:
        """A failed attach may have left a half-spawned publisher behind."""
        built: list[_FakeCamera] = []

        def build() -> _FakeCamera:
            camera = _FakeCamera()
            if len(built) == 0:
                camera.connect_error = RuntimeError("timed out")
            built.append(camera)
            return camera

        connect_with_retry(build, name="overhead", attempts=2, backoff=0)

        assert len(built) == 2
        assert built[0] is not built[1]

    def test_one_attempt_means_no_retry(self) -> None:
        builds = 0

        def build() -> _FakeCamera:
            nonlocal builds
            builds += 1
            camera = _FakeCamera()
            camera.connect_error = RuntimeError("timed out")
            return camera

        with pytest.raises(RuntimeError):
            connect_with_retry(build, name="overhead", attempts=1, backoff=0)

        assert builds == 1


class TestAsAWorker:
    """The worker owns its camera and its thread, not just a tick."""

    async def test_entering_attaches_and_leaving_releases(self) -> None:
        camera = _FakeCamera(frames=[_Frame(np.zeros((480, 640, 3), np.uint8), timestamp=1.0)])
        store = FeatureStore(FeatureSpec.build(camera_features({"overhead": (480, 640, 3)})))
        worker = CameraWorker(lambda: camera, store, key=KEY, hz=100.0)

        async with worker:
            assert camera.connected
            await asyncio.sleep(0.1)
            assert store.read(KEY) is not None, "the worker's own thread never ticked"

        assert not camera.connected
        assert camera.disconnects == 1

    async def test_ticking_outside_acquire_is_refused(self) -> None:
        """Otherwise a tick after unload would read a camera that was let go."""
        camera = _FakeCamera(frames=[_Frame(np.zeros((480, 640, 3), np.uint8), timestamp=1.0)])
        store = FeatureStore(FeatureSpec.build(camera_features({"overhead": (480, 640, 3)})))
        worker = CameraWorker(lambda: camera, store, key=KEY, hz=30.0)

        with pytest.raises(RuntimeError, match="not attached"):
            worker.tick()

        async with worker:
            pass

        with pytest.raises(RuntimeError, match="not attached"):
            worker.tick()

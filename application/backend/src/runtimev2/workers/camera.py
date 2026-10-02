"""Own one camera and publish its frames into the store.

This is the only place in ``runtimev2`` that knows cameras are served over
iceoryx2. Everything downstream -- a recording, a model, the websocket -- reads
frames out of the store like any other feature, so none of it has to handle a
publisher that died or a device that is busy.

The worker ticks at the camera's own rate, independently of the robots and of
whatever is recording. That is the point of the store: a 30Hz camera, a 100Hz
arm and a 50Hz recording need no common clock, because every value carries the
time it was measured.

The environment's declared resolution leads. A publisher serving something else
is resized to match rather than refused, because the stream size is not always
ours to choose -- an IP camera serves what it serves -- and the declared shape
is what the feature spec promised, what a model was trained on, and what a
dataset already holds.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

import cv2
import numpy as np
from loguru import logger

from runtimev2.workers.base import ThreadedWorker

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from physicalai.capture.camera import Camera

    from runtimev2.store import FeatureStore

_COLOR_CHANNELS = 3
_GRAYSCALE_DIMS = 2

# A device the kernel has only just taken back from another publisher is not
# ready to stream: its driver times out waiting for a first frame, with a
# timeout the publisher sets and we cannot pass through. Attaching again a
# moment later works, so the attach is retried rather than the load failing.
CONNECT_ATTEMPTS = 3
CONNECT_BACKOFF_S = 1.0


def conform(data: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Return ``data`` as ``shape``, resizing and promoting channels as needed.

    Spatial dimensions are resized; a single-channel frame is promoted to three
    first, so a grayscale or depth publisher fills a colour feature with
    something coherent rather than a differently shaped array. A frame that is
    already the right shape is returned untouched, which is the normal case.

    Args:
        data: The publisher's frame.
        shape: The ``(h, w, c)`` the environment declared.

    Returns:
        An array of exactly ``shape``.

    Raises:
        ValueError: ``data`` has a channel count that cannot be conformed, or
            ``shape`` asks for something other than colour.
    """
    height, width, channels = shape
    if channels != _COLOR_CHANNELS:
        raise ValueError(f"camera features are colour; cannot produce {channels} channels")

    if data.ndim == _GRAYSCALE_DIMS:
        data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
    elif data.ndim != _COLOR_CHANNELS or data.shape[2] != _COLOR_CHANNELS:
        raise ValueError(f"cannot conform a frame of shape {data.shape} to {shape}")

    if data.shape[:2] != (height, width):
        # cv2 takes (w, h); every other shape in this codebase is (h, w).
        data = cv2.resize(data, (width, height))
    return data


def connect_with_retry(
    build: Callable[[], Camera],
    *,
    name: str,
    attempts: int = CONNECT_ATTEMPTS,
    backoff: float = CONNECT_BACKOFF_S,
) -> Camera:
    """Attach to a camera, retrying while the device settles.

    A fresh camera is built per attempt rather than reconnecting the same one:
    a failed attach may have left a half-spawned publisher behind, and starting
    clean is cheaper to reason about than unpicking that.

    Blocking, so call it off the event loop.

    Args:
        build: Makes an unconnected camera. Called once per attempt.
        name: For the log line when an attempt fails.
        attempts: How many times to try. One means no retry.
        backoff: Seconds between attempts.

    Returns:
        The connected camera.

    Raises:
        Exception: Whatever the last attempt raised, once they are used up. The
            final failure is the one worth reporting -- an earlier one is just
            a device that had not woken up.
    """
    for attempt in range(1, attempts + 1):
        camera = build()
        try:
            camera.connect()
        except Exception as exc:
            if attempt == attempts:
                raise
            logger.warning(
                "Camera {} did not attach on attempt {}/{} ({}); retrying in {}s",
                name,
                attempt,
                attempts,
                exc,
                backoff,
            )
            time.sleep(backoff)
        else:
            if attempt > 1:
                logger.info("Camera {} attached on attempt {}", name, attempt)
            return camera
    raise AssertionError("unreachable: the loop either returns or raises")


class CameraWorker(ThreadedWorker):
    """Hold one camera open and write its latest frame to the store.

    Takes a builder rather than a camera because attaching is retried, and each
    attempt needs a fresh one. ``RobotWorker`` takes its device directly, which
    is the difference between a device worth retrying and one that is not.
    """

    def __init__(
        self,
        build: Callable[[], Camera],
        store: FeatureStore,
        *,
        key: str,
        hz: float,
    ) -> None:
        super().__init__(name=key, hz=hz)
        self._build = build
        self._store = store
        self._key = key
        declared = store.spec[key].shape
        if len(declared) != _COLOR_CHANNELS:
            raise ValueError(f"{key!r} is not an image feature")
        # Unpacked rather than kept as a variable-length tuple, so the declared
        # shape is a ``(h, w, c)`` from here on.
        height, width, channels = declared
        self._shape: tuple[int, int, int] = (height, width, channels)
        self._camera: Camera | None = None

    @property
    def key(self) -> str:
        """The image feature this worker authors."""
        return self._key

    @contextmanager
    def acquire(self) -> Generator[None]:
        """Attach to the camera's publisher, and let go of it after.

        Runs on this worker's thread, so spawning the publisher process and
        waiting for its first frame block here rather than on the event loop.
        Retried, because a device just released by another publisher needs a
        moment before it will stream.
        """
        camera = connect_with_retry(self._build, name=self._key)
        self._camera = camera
        try:
            yield
        finally:
            self._camera = None
            camera.disconnect()

    def tick(self) -> None:
        """Publish the newest frame, at the resolution the environment declared.

        ``read_latest`` returns whatever the publisher has now rather than
        waiting for the next frame, so a worker ticking faster than the camera
        republishes the same frame with the same timestamp. A reader comparing
        timestamps can see that; a reader that only wants the current image does
        not have to care.

        Frames are already copies out of shared memory -- the camera is built
        without zero-copy -- so what goes into the store is safe to hand to
        several readers.

        Raises:
            RuntimeError: Ticked outside ``acquire``.
        """
        if self._camera is None:
            raise RuntimeError(f"Camera {self._key} is not attached")
        frame = self._camera.read_latest()
        data = conform(np.asarray(frame.data), self._shape)
        self._store.write(self._key, data, timestamp=frame.timestamp)

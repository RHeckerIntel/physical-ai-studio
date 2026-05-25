"""Process-local registry for CameraWorker instances with ref-counting.

Ensures that only one CameraWorker process runs per camera fingerprint at a
time, regardless of how many callers request the same camera concurrently.
"""

import asyncio
import threading
from dataclasses import dataclass, field
from multiprocessing.synchronize import Event as EventClass

from schemas.project_camera import Camera
from workers.camera_worker import CameraWorker

_lock = threading.Lock()


@dataclass
class _Entry:
    worker: CameraWorker
    refcount: int = field(default=1)


_registry: dict[str, _Entry] = {}


async def acquire_camera_worker(
    camera: Camera,
    mp_stop_event: EventClass,
    *,
    load_timeout_s: float = 10.0,
) -> CameraWorker:
    """Return a started CameraWorker for *camera*, sharing an existing one when possible.

    If a live worker is already registered for ``camera.fingerprint`` its
    reference count is incremented and the existing instance is returned.
    Otherwise a new worker is created, started, and stored with refcount=1.

    The caller **must** pair every successful call with a corresponding
    :func:`release_camera_worker` call.

    Raises:
        RuntimeError: If the worker does not finish loading within
            *load_timeout_s* seconds.
    """
    fingerprint = camera.fingerprint

    with _lock:
        entry = _registry.get(fingerprint)
        if entry is not None:
            entry.refcount += 1
            return entry.worker

        worker = CameraWorker(camera, mp_stop_event)
        worker.start()
        _registry[fingerprint] = _Entry(worker=worker, refcount=1)

    # Wait for the worker to finish loading outside the lock so we don't
    # block other threads while the camera initialises.
    loaded = await asyncio.to_thread(worker.loaded_event.wait, load_timeout_s)
    if not loaded:
        release_camera_worker(fingerprint)
        raise RuntimeError(f"CameraWorker for fingerprint '{fingerprint}' did not load within {load_timeout_s}s")

    return worker


def release_camera_worker(fingerprint: str) -> None:
    """Decrement the reference count for *fingerprint*.

    When the count reaches zero the worker is removed from the registry and
    stopped.  ``worker.stop()`` is intentionally called **outside** the lock
    to keep the lock scope minimal.
    """
    worker_to_stop: CameraWorker | None = None

    with _lock:
        entry = _registry.get(fingerprint)
        if entry is None:
            return
        entry.refcount -= 1
        if entry.refcount <= 0:
            worker_to_stop = entry.worker
            del _registry[fingerprint]

    if worker_to_stop is not None:
        worker_to_stop.stop()

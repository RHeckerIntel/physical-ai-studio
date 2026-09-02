import threading
from multiprocessing.synchronize import Event as EventClass

import cv2
from loguru import logger

from schemas.project_camera import Camera
from utils.camera_factory import build_shared_camera
from utils.jpeg import encode_jpeg_rgb
from workers.base import BaseThreadWorker, run_at_frequency


class CameraWorker(BaseThreadWorker):
    """Orchestrates camera streaming over configurable transport.

    Capture and JPEG encoding happen on this worker's own thread so the
    shared FastAPI event loop only ever sends already-encoded bytes; encoding
    there would block every other websocket connection (cameras and the
    runtime session stream) while it runs, since the app is single-process.
    """

    def __init__(
        self,
        config: Camera,
        stop_event: EventClass,
        is_locked: bool = False,
    ) -> None:
        super().__init__(stop_event=stop_event)

        # TODO explicitly add width, height to ip camera
        self._width = config.payload.width or 640
        self._height = config.payload.height or 480
        self.camera = build_shared_camera(
            config=config,
            validate_on_connect=False,
            overwrite_settings=not is_locked,
        )
        self._jpeg_lock = threading.Lock()
        self._jpeg_data: bytes | None = None
        self.config = config

    def get_jpeg_frame(self) -> bytes | None:
        """Return the most recently encoded frame, or None before the first frame arrives."""
        with self._jpeg_lock:
            return self._jpeg_data

    def setup(self) -> None:
        self.camera.connect()

    async def run_loop(self) -> None:
        """Main worker loop."""
        try:
            while not self.should_stop():
                async with run_at_frequency(self.config.payload.fps):
                    frame = self.camera.read_latest()
                    data = frame.data
                    if data.shape[:2] != (self._height, self._width):
                        data = cv2.resize(data, (self._width, self._height))
                    jpeg = encode_jpeg_rgb(data)
                    with self._jpeg_lock:
                        self._jpeg_data = jpeg
        except Exception as e:
            logger.error(e)
        finally:
            logger.info("Camera run loop stopped. Disconnecting")
            if self.camera.is_connected:
                self.camera.disconnect()

    async def teardown(self) -> None:
        await super().teardown()

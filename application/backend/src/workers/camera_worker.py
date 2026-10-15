import ctypes
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from multiprocessing import Array
from typing import Any

import cv2
import numpy as np
from loguru import logger

from schemas.project_camera import Camera
from utils.camera_factory import build_shared_camera
from workers.base import ThreadWorker
from workers.rate import Rate


class CameraWorker(ThreadWorker[None]):
    """Orchestrates camera streaming over configurable transport."""

    def __init__(
        self,
        config: Camera,
        is_locked: bool = False,
    ) -> None:
        super().__init__(name=f"CameraWorker-{config.name}")

        # TODO explicitly add width, height to ip camera
        self._width = config.payload.width or 640
        self._height = config.payload.height or 480
        self.camera = build_shared_camera(
            config=config,
            validate_on_connect=False,
            overwrite_settings=not is_locked,
        )
        self._frame_data = Array(ctypes.c_uint8, self._width * self._height * 3)
        self.config = config

    def get_frame(self) -> np.ndarray:
        with self._frame_data.get_lock():
            return self.frame_from_buffer(self._frame_data.get_obj(), self._width, self._height)

    @staticmethod
    def frame_from_buffer(buffer: Any, width: int, height: int) -> np.ndarray:
        return np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3).copy()

    def _set_frame(self, data: np.ndarray) -> None:
        if data.shape[:2] != (self._height, self._width):
            data = cv2.resize(data, (self._width, self._height))
        with self._frame_data.get_lock():
            np.frombuffer(self._frame_data.get_obj(), dtype=np.uint8)[:] = data.reshape(-1)

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[None]:
        self.camera.connect()
        try:
            yield
        finally:
            logger.info("Camera run loop stopped. Disconnecting")
            if self.camera.is_connected:
                self.camera.disconnect()

    async def run_loop(self, context: None) -> None:  # noqa: ARG002
        rate = Rate(self.config.payload.fps, name=self.name)
        while not self.should_stop():
            frame = self.camera.read_latest()
            self._set_frame(frame.data)
            await rate.sleep()

import asyncio
import ctypes
from multiprocessing import Array
from unittest.mock import MagicMock, patch

import numpy as np

from workers.camera_worker import CameraWorker


def _make_config(width=640, height=480, fps=30):
    from schemas.project_camera import CameraAdapter

    return CameraAdapter.validate_python(
        {
            "driver": "usb_camera",
            "name": "test_cam",
            "fingerprint": {"path": "/dev/video0"},
            "hardware_name": None,
            "payload": {"width": width, "height": height, "fps": fps},
        }
    )


def _make_frame(height: int = 480, width: int = 640):
    mock_frame = MagicMock()
    mock_frame.data = np.zeros((height, width, 3), dtype=np.uint8)
    return mock_frame


def _make_camera_worker(config, **kwargs):
    return CameraWorker(config, **kwargs)


class TestCameraWorker:
    def test_build_shared_camera_called(self):
        config = _make_config()
        with patch("workers.camera_worker.build_shared_camera") as mock_build:
            mock_build.return_value = MagicMock()
            worker = _make_camera_worker(config)
            mock_build.assert_called_once_with(config=config, validate_on_connect=False, overwrite_settings=True)
            assert worker.camera is not None

    def test_build_shared_camera_called_when_locked(self):
        config = _make_config()
        with patch("workers.camera_worker.build_shared_camera") as mock_build:
            mock_build.return_value = MagicMock()
            _make_camera_worker(config, is_locked=True)
            mock_build.assert_called_once_with(config=config, validate_on_connect=False, overwrite_settings=False)

    def test_get_frame_returns_zeros_initially(self):
        config = _make_config()
        with patch("workers.camera_worker.build_shared_camera") as mock_build:
            mock_build.return_value = MagicMock()
            worker = _make_camera_worker(config)
            frame = worker.get_frame()
            assert frame.shape == (480, 640, 3)
            assert frame.dtype == np.uint8
            assert np.all(frame == 0)

    def test_set_frame_stores_in_buffer(self):
        config = _make_config()
        with patch("workers.camera_worker.build_shared_camera") as mock_build:
            mock_build.return_value = MagicMock()
            worker = _make_camera_worker(config)
            data = np.ones((480, 640, 3), dtype=np.uint8) * 42
            worker._set_frame(data)
            np.testing.assert_array_equal(worker.get_frame(), data)

    def test_set_frame_resizes_if_wrong_shape(self):
        config = _make_config()
        with patch("workers.camera_worker.build_shared_camera") as mock_build:
            mock_build.return_value = MagicMock()
            worker = _make_camera_worker(config)
            worker._set_frame(np.zeros((240, 320, 3), dtype=np.uint8))
            assert worker.get_frame().shape == (480, 640, 3)

    def test_frame_from_buffer(self):
        buf = Array(ctypes.c_uint8, 640 * 480 * 3)
        np.frombuffer(buf.get_obj(), dtype=np.uint8)[:] = 7
        frame = CameraWorker.frame_from_buffer(buf.get_obj(), 640, 480)
        assert frame.shape == (480, 640, 3)
        assert np.all(frame == 7)

    def test_should_stop_reflects_stop_event(self):
        config = _make_config()
        with patch("workers.camera_worker.build_shared_camera") as mock_build:
            mock_build.return_value = MagicMock()
            worker = _make_camera_worker(config)
            assert not worker.should_stop()
            worker.request_stop()
            assert worker.should_stop()

    def test_lifecycle_connects_and_disconnects_camera(self):
        config = _make_config()
        with patch("workers.camera_worker.build_shared_camera") as mock_build:
            mock_cam = MagicMock()
            mock_cam.is_connected = True
            mock_build.return_value = mock_cam
            worker = _make_camera_worker(config)

            async def enter_and_leave() -> None:
                async with worker.lifecycle():
                    mock_cam.connect.assert_called_once()
                    mock_cam.disconnect.assert_not_called()

            asyncio.run(enter_and_leave())
            mock_cam.disconnect.assert_called_once()

    def test_lifecycle_skips_disconnect_if_not_connected(self):
        config = _make_config()
        with patch("workers.camera_worker.build_shared_camera") as mock_build:
            mock_cam = MagicMock()
            mock_cam.is_connected = False
            mock_build.return_value = mock_cam
            worker = _make_camera_worker(config)

            async def enter_and_leave() -> None:
                async with worker.lifecycle():
                    pass

            asyncio.run(enter_and_leave())
            mock_cam.disconnect.assert_not_called()

    def test_run_loop_reads_frames_until_stopped(self):
        config = _make_config()
        with patch("workers.camera_worker.build_shared_camera") as mock_build:
            mock_cam = MagicMock()
            mock_build.return_value = mock_cam
            worker = _make_camera_worker(config)
            call_count = 0

            def read_latest():
                nonlocal call_count
                call_count += 1
                if call_count >= 3:
                    worker.request_stop()
                return _make_frame()

            mock_cam.read_latest.side_effect = read_latest
            asyncio.run(worker.run_loop(None))

            assert call_count == 3

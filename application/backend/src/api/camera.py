import json
from typing import Annotated

from fastapi import APIRouter, Depends, Query, WebSocket
from fastapi.responses import Response
from fastapi.websockets import WebSocketDisconnect
from frame_source import FrameSourceFactory

from api.dependencies import SchedulerDep
from schemas.camera import SupportedCameraFormat
from schemas.project_camera import Camera as ProjectCamera
from schemas.project_camera import CameraAdapter
from workers.base import run_at_frequency
from workers.camera_worker_registry import acquire_camera_worker, release_camera_worker

router = APIRouter(prefix="/api/cameras", tags=["Cameras"])


@router.get("/supported_formats/{driver}")
async def get_supported_formats(
    driver: str,
    fingerprint: str,
) -> list[SupportedCameraFormat]:
    """Returns the supported camera resolution and fps associated to the camera"""
    camera = FrameSourceFactory.create(driver if driver != "usb_camera" else "webcam", source=fingerprint)
    formats = camera.get_supported_formats()

    if formats is None:
        return []

    return [
        SupportedCameraFormat(width=format["width"], height=format["height"], fps=format["fps"]) for format in formats
    ]


def get_camera_from_query(websocket: WebSocket) -> ProjectCamera:
    """Parse camera from query parameters."""
    camera_param = websocket.query_params.get("camera")
    if not camera_param:
        raise ValueError("Missing 'camera' query parameter")

    try:
        camera_data = json.loads(camera_param)
        return CameraAdapter.validate_python(camera_data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in camera parameter: {e}")
    except Exception as e:
        raise ValueError(f"Invalid camera configuration: {e}")


@router.get("/ws", tags=["WebSocket"], summary="Camera streaming (WebSocket)", status_code=426)
async def camera_websocket_openapi(
    camera: Annotated[str | None, Query(description="JSON-serialized ProjectCamera configuration")] = None,  # noqa: ARG001
) -> Response:
    """This endpoint requires a WebSocket connection. Use `wss://` to connect."""
    return Response(status_code=426)


@router.websocket("/ws")
async def camera_websocket(
    websocket: WebSocket,
    scheduler: SchedulerDep,
    camera: Annotated[ProjectCamera, Depends(get_camera_from_query)],
) -> None:
    """
    WebSocket endpoint for camera streaming.

    Query Parameters:
        camera: JSON serialized ProjectCamera

    Protocol:
        Client sends JSON messages:
            {"event": "disconnect"} - Request graceful disconnect
            {"event": "ping"} - Keep-alive check

        Server sends JSON-encoded messages with status updates:
            {"event": "status", "state": "running", ...}
    """
    import cv2

    await websocket.accept()

    worker = None
    try:
        worker = await acquire_camera_worker(camera, scheduler.mp_stop_event)
        while True:
            async with run_at_frequency(camera.payload.fps):
                frame = worker.get_frame()
                success, jpeg = cv2.imencode(".jpg", frame)
                if success and jpeg is not None:
                    await websocket.send_bytes(jpeg.tobytes())
    except WebSocketDisconnect:
        pass
    finally:
        if worker is not None:
            release_camera_worker(camera.fingerprint)

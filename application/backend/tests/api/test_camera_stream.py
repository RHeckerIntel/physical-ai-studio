from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import quote
from uuid import uuid4

from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect

from api.dependencies import get_camera_claim_registry, get_scheduler
from main import app
from services.camera_claims import CameraClaim, CameraClaimRegistry


def _camera_query(*, fingerprint: dict[str, str] | None = None) -> str:
    return json.dumps(
        {
            "driver": "usb_camera",
            "name": "front",
            "fingerprint": fingerprint or {"serial": "cam-front"},
            "hardware_name": None,
            "payload": {"width": 640, "height": 480, "fps": 30},
        }
    )


def _client(claims: CameraClaimRegistry) -> TestClient:
    scheduler = MagicMock()
    app.dependency_overrides[get_scheduler] = lambda: scheduler
    app.dependency_overrides[get_camera_claim_registry] = lambda: claims
    return TestClient(app)


def test_stream_endpoint_passes_is_locked_for_a_claimed_fingerprint() -> None:
    claims = CameraClaimRegistry()
    claims.claim(
        [
            CameraClaim(
                fingerprint={"serial": "cam-front"},
                settings=(640, 480, 30),
                holder="rt-a",
                project_id=uuid4(),
                project_name="Alpha",
            )
        ]
    )
    client = _client(claims)
    try:
        with (
            patch("api.camera.build_shared_camera") as build,
            patch("api.camera.Rate") as rate_cls,
            patch("utils.jpeg.encode_jpeg_rgb", return_value=b"jpeg"),
        ):
            rate_cls.return_value.sleep = AsyncMock(side_effect=WebSocketDisconnect())
            with client.websocket_connect(f"/api/cameras/ws?camera={quote(_camera_query())}"):
                pass
            assert build.call_args.kwargs["overwrite_settings"] is False
            build.return_value.disconnect.assert_called_once()
    finally:
        app.dependency_overrides.clear()


def test_stream_endpoint_is_unlocked_when_the_camera_is_unclaimed() -> None:
    client = _client(CameraClaimRegistry())
    try:
        with (
            patch("api.camera.build_shared_camera") as build,
            patch("api.camera.Rate") as rate_cls,
            patch("utils.jpeg.encode_jpeg_rgb", return_value=b"jpeg"),
        ):
            rate_cls.return_value.sleep = AsyncMock(side_effect=WebSocketDisconnect())
            with client.websocket_connect(f"/api/cameras/ws?camera={quote(_camera_query())}"):
                pass
            assert build.call_args.kwargs["overwrite_settings"] is True
    finally:
        app.dependency_overrides.clear()

"""Pinning stops two sessions disagreeing about how a camera is configured."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import pytest

from exceptions import CameraSettingsConflictError
from runtimev2.pinning import CameraPinning
from services.camera_claims import CameraClaimRegistry

PROJECT_ID = uuid4()


@dataclass
class _Payload:
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class _Camera:
    name: str = "overhead"
    fingerprint: dict[str, Any] | None = None
    payload: _Payload = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.fingerprint is None and self.name != "unidentified":
            self.fingerprint = {"serial": "SERIAL-1"}
        if self.payload is None:
            self.payload = _Payload()


def _pinning(registry: CameraClaimRegistry) -> CameraPinning:
    return CameraPinning(registry=registry, project_id=PROJECT_ID, project_name="Demo")


class TestPinning:
    def test_sharing_a_camera_at_the_same_settings_is_allowed(self) -> None:
        """Two sessions can watch one camera; the publisher is built for it."""
        registry = CameraClaimRegistry()

        with _pinning(registry).hold([_Camera()]), _pinning(registry).hold([_Camera()]):
            pass

    def test_a_second_session_wanting_other_settings_is_refused(self) -> None:
        """Otherwise it would silently receive the first session's frames."""
        registry = CameraClaimRegistry()
        other = _Camera(payload=_Payload(width=1280, height=720, fps=30))

        with ExitStack() as other_session:
            other_session.enter_context(_pinning(registry).hold([_Camera()]))

            with pytest.raises(CameraSettingsConflictError) as caught, _pinning(registry).hold([other]):
                pass

        assert "Demo" in caught.value.message

    def test_settings_are_released_with_the_block(self) -> None:
        registry = CameraClaimRegistry()
        other = _Camera(payload=_Payload(width=1280, height=720, fps=30))

        with _pinning(registry).hold([_Camera()]):
            pass

        with _pinning(registry).hold([other]):
            pass

    def test_an_unidentified_camera_is_refused(self) -> None:
        """Without a fingerprint there is nothing to pin it against."""
        registry = CameraClaimRegistry()
        camera = _Camera(name="unidentified")
        camera.fingerprint = None

        with pytest.raises(ValueError, match="must be reselected"), _pinning(registry).hold([camera]):
            pass

    def test_no_cameras_is_not_a_conflict(self) -> None:
        with _pinning(CameraClaimRegistry()).hold([]):
            pass

    def test_each_session_pins_under_its_own_holder(self) -> None:
        """So releasing one cannot unpin another that reused the name."""
        registry = CameraClaimRegistry()

        assert _pinning(registry).holder != _pinning(registry).holder

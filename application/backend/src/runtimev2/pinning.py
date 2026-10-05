"""Pin the camera settings a session needs, so a clash is an error.

A publisher process serves one resolution and framerate. A second session
asking for different ones is handed the first session's instead, and this
runtime resizes whatever arrives to the resolution its environment declares --
which would make the mismatch invisible, and a recording labelled 1280x720
would hold upscaled 640x480 pixels.

Pinning turns that into a refusal at load. It does not stop two sessions
sharing a camera, which is what the publisher is for; it stops them disagreeing
about how it is configured.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import uuid4

from services.camera_claims import CameraClaim, settings_from_camera

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from uuid import UUID

    from schemas.project_camera import Camera
    from services.camera_claims import CameraClaimRegistry


@dataclass(frozen=True, slots=True)
class CameraPinning:
    """What a session needs in order to pin its cameras.

    Attributes:
        registry: Process-local registry shared with every other session,
            including the old runtime's, so the two cannot disagree either.
        project_id: Reported when another session is already holding a camera.
        project_name: Reported in the same message, which is what makes a
            conflict actionable rather than puzzling.
        holder: Unique per session, so releasing one cannot unpin another.
    """

    registry: CameraClaimRegistry
    project_id: UUID
    project_name: str
    holder: str = field(default_factory=lambda: f"rtv2-{uuid4().hex}")

    @contextmanager
    def hold(self, cameras: Sequence[Camera]) -> Iterator[None]:
        """Pin ``cameras`` for the body of the block, releasing them after.

        Raises:
            CameraSettingsConflictError: Another session pinned other settings.
            ValueError: A camera has no fingerprint, so it cannot be identified
                and therefore cannot be pinned.
        """
        unidentified = [camera.name for camera in cameras if camera.fingerprint is None]
        if unidentified:
            raise ValueError(f"Camera {', '.join(unidentified)} must be reselected before it can be used")
        claims = [
            CameraClaim(
                fingerprint=camera.fingerprint,
                settings=settings_from_camera(camera),
                holder=self.holder,
                project_id=self.project_id,
                project_name=self.project_name,
            )
            for camera in cameras
            if camera.fingerprint is not None
        ]
        with self.registry.hold(claims):
            yield

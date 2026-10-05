"""Pin the camera settings a session needs, so a clash is an error.

A publisher serves one resolution. A second session asking for another is
handed the first's, and this runtime resizes to whatever its environment
declares -- so a recording labelled 1280x720 would hold upscaled 640x480
pixels, invisibly. Pinning makes that a refusal at load.

It does not stop two sessions sharing a camera, which is what the publisher is
for. It stops them disagreeing about how it is configured.
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
        registry: Shared with every other session, including the old runtime's.
        project_id: Reported when another session already holds a camera.
        project_name: Reported with it, which makes a conflict actionable.
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
            ValueError: A camera has no fingerprint to be pinned against.
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

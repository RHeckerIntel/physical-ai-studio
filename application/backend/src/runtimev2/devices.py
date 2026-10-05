"""The devices a session runs, named, with the rows they came from.

A saved environment is one way to say which devices to use; the setup wizard is
another, verifying a robot registered moments ago with no environment to load.
Both reduce to robots to drive, robots to read, cameras, and a name.

Rows rather than ids, because building a driver needs a robot's type and port
and opening a camera needs its fingerprint and resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from schemas.environment import EnvironmentWithRelations
    from schemas.project_camera import Camera
    from schemas.robot import ReadableRobot


@dataclass(frozen=True, slots=True)
class DeviceSet:
    """What to open, and what to call it.

    Attributes:
        name: Shown to a client and used in logs.
        robots: Robots this session drives.
        leaders: Robots it only reads -- separate because nothing commands
            them, which is why they contribute no action features.
        cameras: Cameras to open, at the resolution their rows declare.
    """

    name: str
    robots: tuple[ReadableRobot, ...] = ()
    leaders: tuple[ReadableRobot, ...] = ()
    cameras: tuple[Camera, ...] = ()

    def robot_row(self, robot_id: str) -> ReadableRobot:
        """Find a robot row by id, driven or read.

        Raises:
            RuntimeError: No such robot in this set.
        """
        for candidate in (*self.robots, *self.leaders):
            if str(candidate.id) == robot_id:
                return candidate
        raise RuntimeError(f"Robot {robot_id} is not part of {self.name}")

    def camera_row(self, camera_id: str) -> Camera:
        """Find a camera row by id.

        Raises:
            RuntimeError: No such camera in this set.
        """
        for candidate in self.cameras:
            if str(candidate.id) == camera_id:
                return candidate
        raise RuntimeError(f"Camera {camera_id} is not part of {self.name}")


def from_environment(environment: EnvironmentWithRelations) -> DeviceSet:
    """Read a device set out of a saved environment.

    A configured robot's teleoperator becomes a leader: that is the pairing.
    """
    robots: list[ReadableRobot] = []
    leaders: list[ReadableRobot] = []
    for configured in environment.robots:
        robots.append(configured.robot)
        teleoperator = getattr(configured.tele_operator, "robot", None)
        if teleoperator is not None:
            leaders.append(teleoperator)
    return DeviceSet(
        name=environment.name,
        robots=tuple(robots),
        leaders=tuple(leaders),
        cameras=tuple(environment.cameras),
    )

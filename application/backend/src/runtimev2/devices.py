"""The devices a session runs, named, with the rows they came from.

A saved environment is one way to say which devices to use. The setup wizard is
another: it verifies a robot that was registered moments ago, so there is no
environment to load and there cannot be one. Both reduce to the same thing --
some robots to drive, some to read, some cameras, and a name to call them by --
so the runtime takes that rather than an environment row.

Rows and not ids, because describing a robot needs its type and port to build a
driver, and opening a camera needs its fingerprint and declared resolution.
Whoever resolves the ids has the services to do it; this is what they produce.
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
        name: Shown to a client and used in logs. An environment's name, or
            something descriptive for an unsaved set.
        robots: Robots this session drives.
        leaders: Robots it only reads. Separate because nothing commands them,
            which is also why they contribute no action features.
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

    A configured robot's teleoperator becomes a leader, because that is what
    the pairing in an environment means.
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

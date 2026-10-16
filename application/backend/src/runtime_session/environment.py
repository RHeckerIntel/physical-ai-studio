"""What loading an environment needs: its store spec, the follower recipe and its cameras.

The joints come from the driver, not the database: a robot row stores a type
and a port, and only the driver for that type knows its joints and their order.
Building a driver does not touch the device; only ``connect()`` does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from robots.features import sanitize_camera_name
from runtime_session.store import FeatureSpec
from workers.camera_worker import camera_image_spec
from workers.robot_worker import RobotRecipe

if TYPE_CHECKING:
    from collections.abc import Mapping

    from robots.robot_client_factory import RobotClientFactory
    from schemas.environment import EnvironmentWithRelations
    from schemas.project_camera import Camera


@dataclass(frozen=True, slots=True)
class EnvironmentDescription:
    spec: FeatureSpec
    follower: RobotRecipe
    cameras: Mapping[str, Camera]
    """Camera rows by store key, the key a dataset records them under."""


async def describe_environment(
    environment: EnvironmentWithRelations, factory: RobotClientFactory
) -> EnvironmentDescription:
    """Build the follower's driver (unconnected) and derive the session's spec.

    Raises:
        ValueError: The environment does not have exactly one robot, or two
            cameras share a store key.
    """
    if len(environment.robots) != 1:
        raise ValueError(
            f"Environment {environment.name!r} has {len(environment.robots)} robots; "
            "a session drives exactly one robot for now"
        )
    robot = environment.robots[0].robot
    driver, _definition = await factory.build_robot_driver(robot, factory)
    follower = RobotRecipe.from_driver(robot.id, driver)

    cameras: dict[str, Camera] = {}
    for camera in environment.cameras:
        key = sanitize_camera_name(camera.name)
        if key in cameras:
            raise ValueError(f"Cameras {cameras[key].name!r} and {camera.name!r} would both be recorded as {key!r}")
        cameras[key] = camera
    spec = FeatureSpec(follower.joint_names, {key: camera_image_spec(camera) for key, camera in cameras.items()})
    return EnvironmentDescription(spec=spec, follower=follower, cameras=cameras)

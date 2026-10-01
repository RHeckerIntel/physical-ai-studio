"""Read a session's shape out of an environment, without touching hardware.

What features a robot has is the driver's answer, not the database's: a robot
row stores a type and a port, and only the driver for that type knows which
joints it has and in what order. So the shape has to come through the driver
layer -- but it does not need a connected one. Constructing a driver is free;
only ``connect()`` touches the device.

That matters because the shape is needed before anything is plugged in. Asking
whether a model or a dataset fits an environment is a question the UI should
answer while the user is still choosing, not one that fails at connect time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from runtimev2.features import FeatureSpec, camera_features, robot_features, sanitize_name

if TYPE_CHECKING:
    from physicalai_studio_plugin import CatalogRobotFactory, SerialPortInfo

    from robots.robot_client_factory import RobotClientFactory
    from schemas.environment import EnvironmentWithRelations
    from schemas.project_camera import Camera
    from schemas.robot import ReadableRobot


class _StoredPortFinder:
    """Port finder that falls back to the port a robot was registered with.

    Describing an environment must work with nothing attached, and the real
    finder returns ``None`` for an absent device, which the catalog builders
    turn into an error. The stored path is good enough to construct a driver
    and read its joints off it; it is never used to connect, so the usual
    objection -- that the path may now belong to a different device -- does not
    apply. ``runtime.config_builder`` does the same for exports.
    """

    def __init__(self, port_finder: CatalogRobotFactory) -> None:
        self._port_finder = port_finder

    async def find_port(self, port_info: SerialPortInfo) -> str | None:
        port = await self._port_finder.find_port(port_info)
        return port if port is not None else port_info.connection_string


@dataclass(frozen=True, slots=True)
class RobotShape:
    """One robot in an environment, and the joints its driver reports."""

    key: str
    """Name this robot's features are keyed by. Session-local, see ``describe``."""
    robot_id: str
    role: str
    """``follower`` or ``leader``, as the catalog defines it."""
    joint_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CameraShape:
    """One camera in an environment, at the resolution its row declares."""

    key: str
    """Session key: strict, because it also has to be a zenoh key expression."""
    name: str
    """Display name. Kept because the dataset key sanitizes differently -- see
    ``dataset_layout`` -- and deriving one from the other is not possible."""
    camera_id: str
    shape: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class EnvironmentShape:
    """Everything an environment contributes to a session's features."""

    robots: tuple[RobotShape, ...]
    cameras: tuple[CameraShape, ...]

    def feature_spec(self) -> FeatureSpec:
        """Project this shape onto the session's feature keys."""
        return FeatureSpec.build(
            *(robot_features(robot.key, robot.joint_names) for robot in self.robots),
            camera_features({camera.key: camera.shape for camera in self.cameras}),
        )

    @property
    def followers(self) -> tuple[RobotShape, ...]:
        return tuple(robot for robot in self.robots if robot.role == "follower")


def _camera_shape(camera: Camera) -> tuple[int, int, int]:
    """Return the ``(h, w, c)`` a camera row declares.

    Declared, not actual: a publisher already serving the device at another
    resolution wins at connect time, and a recording stores what the session
    saw. This is the shape to check a model against before connecting, and the
    one to re-check once a frame has arrived.
    """
    payload = camera.payload
    height = int(getattr(payload, "height", None) or 0)
    width = int(getattr(payload, "width", None) or 0)
    return (height, width, 3)


async def describe_environment(
    environment: EnvironmentWithRelations,
    factory: RobotClientFactory,
) -> EnvironmentShape:
    """Return an environment's shape, with nothing attached.

    Robot feature keys come from the robot's display name. That is editable, so
    the keys are session-local only -- which is safe because they never reach
    disk: a recorded dataset stores joint names packed into ``action`` and
    ``observation.state``, with no robot prefix at all. Renaming a robot
    therefore cannot invalidate a dataset.

    Raises:
        RobotPluginUnavailableError: A robot's plugin is not installed.
        ValueError: A robot's type is not in the catalog, or two devices
            collide on a feature key.
    """
    port_finder = _StoredPortFinder(factory)
    robots: list[RobotShape] = []
    for configured in environment.robots:
        robots.append(await _describe_robot(configured.robot, factory, port_finder))
        teleoperator = getattr(configured.tele_operator, "robot", None)
        if teleoperator is not None:
            robots.append(await _describe_robot(teleoperator, factory, port_finder))

    cameras = [
        CameraShape(
            key=sanitize_name(camera.name),
            name=camera.name,
            camera_id=str(camera.id),
            shape=_camera_shape(camera),
        )
        for camera in environment.cameras
    ]

    _reject_key_collisions(robots, cameras)
    return EnvironmentShape(robots=tuple(robots), cameras=tuple(cameras))


async def _describe_robot(
    robot: ReadableRobot,
    factory: RobotClientFactory,
    port_finder: _StoredPortFinder,
) -> RobotShape:
    """Build a robot's driver far enough to read its joints, then drop it."""
    driver, definition = await factory.build_robot_driver(robot, port_finder)
    joint_names = tuple(str(name) for name in driver.joint_names)
    logger.debug("Robot {} ({}) reports joints {}", robot.name, robot.type, joint_names)
    return RobotShape(
        key=sanitize_name(robot.name),
        robot_id=str(robot.id),
        role=definition.role,
        joint_names=joint_names,
    )


def _reject_key_collisions(robots: list[RobotShape], cameras: list[CameraShape]) -> None:
    """Fail loudly when two devices sanitize to the same feature key.

    Silently merging them would put two robots' joints under one name, and the
    only symptom would be a recording where half the columns move together.
    """
    for label, keys in (("Robot", [robot.key for robot in robots]), ("Camera", [camera.key for camera in cameras])):
        seen: set[str] = set()
        for key in keys:
            if key in seen:
                raise ValueError(f"{label} names collide after sanitizing: {key!r} appears twice")
            seen.add(key)

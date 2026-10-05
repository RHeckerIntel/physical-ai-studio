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

from runtimev2.features import FeatureSpec, camera_features, leader_features, robot_features, sanitize_name

# Only reached when a camera row has no fps recorded; the usual case is that it
# does, because the UI makes you pick a format.
_DEFAULT_CAMERA_FPS = 30.0

if TYPE_CHECKING:
    from physicalai_studio_plugin import CatalogRobotFactory, SerialPortInfo

    from robots.robot_client_factory import RobotClientFactory
    from runtimev2.devices import DeviceSet
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
class LeaderShape:
    """One leader arm: read for its position, never commanded."""

    key: str
    robot_id: str
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
    fps: float
    """The rate its worker ticks at. Each camera runs at its own, independently
    of the robots and of whatever is recording."""


@dataclass(frozen=True, slots=True)
class SessionShape:
    """The devices a session runs, and therefore the data it carries.

    Read out of an environment, but named for the session rather than the
    environment because everything in one conforms to it: the store is keyed by
    it, a dataset can only be appended to if it matches, and a model can only be
    loaded if it fits. It outlives any particular set of connected devices.
    """

    robots: tuple[RobotShape, ...]
    """The robots this session drives."""
    cameras: tuple[CameraShape, ...]
    leaders: tuple[LeaderShape, ...] = ()
    """Input devices, kept separate because nothing commands them."""

    def feature_spec(self) -> FeatureSpec:
        """Project this shape onto the session's feature keys."""
        return FeatureSpec.build(
            *(robot_features(robot.key, robot.joint_names) for robot in self.robots),
            *(leader_features(leader.key, leader.joint_names) for leader in self.leaders),
            camera_features({camera.key: camera.shape for camera in self.cameras}),
        )


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


def _camera_fps(camera: Camera) -> float:
    """Return the rate a camera row declares, falling back to a usable default.

    A row with no fps would otherwise divide by zero in the rate loop. The
    publisher's actual rate may differ; a worker ticking faster simply re-reads
    the same frame, which its timestamp makes visible.
    """
    fps = getattr(camera.payload, "fps", None)
    return float(fps) if fps else _DEFAULT_CAMERA_FPS


async def describe_devices(
    devices: DeviceSet,
    factory: RobotClientFactory,
) -> SessionShape:
    """Return a device set's shape, with nothing attached.

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
    robots = [await _describe_robot(robot, factory, port_finder) for robot in devices.robots]
    leaders = [await _describe_leader(leader, factory, port_finder) for leader in devices.leaders]
    cameras = [
        CameraShape(
            key=sanitize_name(camera.name),
            name=camera.name,
            camera_id=str(camera.id),
            shape=_camera_shape(camera),
            fps=_camera_fps(camera),
        )
        for camera in devices.cameras
    ]

    _reject_key_collisions(robots, leaders, cameras)
    return SessionShape(robots=tuple(robots), cameras=tuple(cameras), leaders=tuple(leaders))


async def _describe_leader(
    robot: ReadableRobot,
    factory: RobotClientFactory,
    port_finder: _StoredPortFinder,
) -> LeaderShape:
    """Describe a leader, which contributes observations and nothing else."""
    driver, _definition = await factory.build_robot_driver(robot, port_finder)
    return LeaderShape(
        key=sanitize_name(robot.name),
        robot_id=str(robot.id),
        joint_names=tuple(str(name) for name in driver.joint_names),
    )


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


def _reject_key_collisions(robots: list[RobotShape], leaders: list[LeaderShape], cameras: list[CameraShape]) -> None:
    """Fail loudly when two devices sanitize to the same feature key.

    Silently merging them would put two robots' joints under one name, and the
    only symptom would be a recording where half the columns move together.
    """
    for label, keys in (
        # Robots and leaders share the observation namespace, so they collide
        # with each other as readily as two robots would.
        ("Robot", [robot.key for robot in robots] + [leader.key for leader in leaders]),
        ("Camera", [camera.key for camera in cameras]),
    ):
        seen: set[str] = set()
        for key in keys:
            if key in seen:
                raise ValueError(f"{label} names collide after sanitizing: {key!r} appears twice")
            seen.add(key)

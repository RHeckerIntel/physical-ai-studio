"""A loaded environment holds the devices, the store and the workers."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator  # noqa: TC003  # used by @asynccontextmanager
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

from runtimev2.control.teleop import TeleopControl
from runtimev2.environment import describe_environment
from runtimev2.features import OBSERVATION_PREFIX, joint_feature_key
from runtimev2.leader import open_leaders
from runtimev2.loaded_environment import LoadedEnvironment

JOINTS = ("shoulder_pan", "gripper")


@dataclass
class _Observation:
    joint_positions: np.ndarray
    timestamp: float
    sensor_data: None = None
    images: None = None

    @property
    def state(self) -> np.ndarray:
        return self.joint_positions


@dataclass
class _Driver:
    """Stands in for a robot driver: the thing the worker owns outright."""

    label: str
    reported_joints: tuple[str, ...] = JOINTS
    position: float = 0.0
    connected: bool = False
    disconnects: int = 0
    reads: int = 0
    sent: list[np.ndarray] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def joint_names(self) -> list[str]:
        # Readable before connecting, which is what the describe step relies on
        # to derive a shape without taking the hardware.
        return list(self.reported_joints)

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False
        self.disconnects += 1

    def get_observation(self) -> _Observation:
        with self.lock:
            self.reads += 1
            return _Observation(
                joint_positions=np.full(len(self.reported_joints), self.position, dtype=np.float32),
                # Advances with reads, as a 100Hz device's clock would. Tying it
                # to position instead would make a still arm look frozen in time,
                # and the gap between readings is what sets a ramp's duration.
                timestamp=100.0 + self.reads * 0.01,
            )

    def send_action(self, action: np.ndarray, **_: Any) -> None:
        with self.lock:
            self.sent.append(np.array(action))


@dataclass
class _Row:
    name: str
    type: str = "SO101"
    id: Any = field(default_factory=uuid4)
    connection_string: str = "/dev/ttyACM0"


@dataclass
class _Teleoperator:
    robot: _Row | None = None


@dataclass
class _Configured:
    robot: _Row
    tele_operator: _Teleoperator = field(default_factory=_Teleoperator)


@dataclass
class _CameraRow:
    name: str
    width: int = 640
    height: int = 480
    fps: int = 30
    id: Any = field(default_factory=uuid4)

    @property
    def payload(self) -> _CameraRow:
        return self


@dataclass
class _FakeSharedCamera:
    """Stands in for a SharedCamera attached to a publisher."""

    shape: tuple[int, int, int] = (480, 640, 3)
    connected: bool = False
    disconnects: int = 0
    connect_error: Exception | None = None
    reads: int = 0

    def connect(self) -> None:
        if self.connect_error is not None:
            raise self.connect_error
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False
        self.disconnects += 1

    def read_latest(self) -> Any:
        import numpy as np

        self.reads += 1
        return SimpleNamespace(data=np.zeros(self.shape, dtype=np.uint8), timestamp=500.0, sequence=0)


@dataclass
class _Environment:
    name: str = "test env"
    robots: list[_Configured] = field(default_factory=list)
    cameras: list[Any] = field(default_factory=list)


@dataclass
class _Definition:
    role: str


@dataclass
class _Factory:
    roles: dict[str, str] = field(default_factory=dict)
    joints: dict[str, tuple[str, ...]] = field(default_factory=dict)
    robots: dict[str, _Driver] = field(default_factory=dict)
    """The driver most recently built per robot -- the one a worker holds."""

    async def build_robot_driver(self, robot: _Row, port_finder: object) -> tuple[Any, _Definition]:
        """Build a fresh driver, as the real factory does on every call.

        Called twice per robot: once to describe the environment and once to
        load it. Keeping the latest is what a worker ends up holding.
        """
        driver = _Driver(robot.name, reported_joints=self.joints.get(robot.name, JOINTS))
        self.robots[robot.name] = driver
        return driver, _Definition(role=self.roles.get(robot.name, "follower"))

    async def find_port(self, port_info: object) -> str | None:
        return None


def _teleop_env() -> tuple[_Environment, _Factory]:
    environment = _Environment(
        robots=[_Configured(robot=_Row("follower"), tele_operator=_Teleoperator(robot=_Row("leader")))]
    )
    return environment, _Factory(roles={"follower": "follower", "leader": "leader"})


def _follower_only_env() -> tuple[_Environment, _Factory]:
    """An environment with nothing to teleoperate from, which is a normal one."""
    environment = _Environment(robots=[_Configured(robot=_Row("follower"), tele_operator=None)])
    return environment, _Factory(roles={"follower": "follower"})


@asynccontextmanager
async def _loaded(environment: _Environment, factory: _Factory, **kwargs: Any) -> AsyncIterator[LoadedEnvironment]:
    """Describe, open the leaders, then load -- the sequence the session runs."""
    async with AsyncExitStack() as stack:
        shape = await describe_environment(environment, factory)  # type: ignore[arg-type]
        leaders = await open_leaders(environment, shape, factory, stack)  # type: ignore[arg-type]
        yield await stack.enter_async_context(
            LoadedEnvironment(environment, factory, shape, leaders, **kwargs)  # type: ignore[arg-type]
        )


async def _settle() -> None:
    """Give the worker threads a few ticks to run."""
    await asyncio.sleep(0.2)


class TestLifecycle:
    async def test_opening_connects_every_robot_and_closing_releases_them(self) -> None:
        environment, factory = _teleop_env()

        async with _loaded(environment, factory) as loaded:
            assert set(loaded.state().robots) == {"follower"}, "a leader is not a driven robot"
            assert set(loaded.state().leaders) == {"leader"}
            assert all(robot.connected for robot in factory.robots.values())

        assert all(not robot.connected for robot in factory.robots.values())
        assert all(robot.disconnects == 1 for robot in factory.robots.values())

    async def test_observations_reach_the_store_on_their_own(self) -> None:
        """No one drives the session; the robot workers tick themselves."""
        environment, factory = _teleop_env()

        async with _loaded(environment, factory) as loaded:
            factory.robots["follower"].position = 2.0
            await _settle()
            sample = loaded.store.read(joint_feature_key(OBSERVATION_PREFIX, "follower", "gripper"))

        assert sample is not None
        assert sample.value == pytest.approx(2.0)

    async def test_a_leader_is_only_read_while_something_uses_it(self) -> None:
        """It has no worker of its own: whichever control drives reads it."""
        environment, factory = _teleop_env()
        leader_gripper = joint_feature_key(OBSERVATION_PREFIX, "leader", "gripper")

        async with _loaded(environment, factory) as loaded:
            factory.robots["leader"].position = 2.0
            await _settle()
            assert loaded.store.read(leader_gripper) is None, "read with nothing using it"

            leader, follower = loaded.pair_for_teleop()
            control = TeleopControl(leader, loaded.store, loaded.action_keys_for(follower), hz=200.0)
            async with control:
                await _settle()
                sample = loaded.store.read(leader_gripper)

        assert sample is not None
        assert sample.value == pytest.approx(2.0)

    async def test_a_device_swapped_after_describing_is_refused(self) -> None:
        """Each build resolves a live port, so a different device on the same
        path would be connected under the other one's shape -- and the action
        vector is assembled in that shape's order, while the arm is live."""
        environment, factory = _teleop_env()
        factory.joints = {"follower": JOINTS, "leader": JOINTS}
        original = factory.build_robot_driver
        builds: dict[str, int] = {}

        async def swap(robot: _Row, port_finder: object) -> tuple[_Driver, _Definition]:
            driver, definition = await original(robot, port_finder)
            builds[robot.name] = builds.get(robot.name, 0) + 1
            if builds[robot.name] > 1:  # the load, after the describe
                driver.reported_joints = ("gripper", "shoulder_pan")  # reversed
            return driver, definition

        factory.build_robot_driver = swap  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="once connected"):
            async with _loaded(environment, factory):
                pass

    async def test_a_robot_connected_before_the_failure_is_still_released(self) -> None:
        """Two robots, the second failing: the first must not be left energized."""
        environment = _Environment(robots=[_Configured(robot=_Row("first")), _Configured(robot=_Row("second"))])
        factory = _Factory(roles={"first": "follower", "second": "follower"})
        original = factory.build_robot_driver
        builds: dict[str, int] = {}

        async def fail_loading_the_second(robot: _Row, port_finder: object) -> tuple[_Driver, _Definition]:
            builds[robot.name] = builds.get(robot.name, 0) + 1
            # The second build is the load; by then the first is connected.
            if robot.name == "second" and builds[robot.name] > 1:
                raise RuntimeError("this driver refused to build")
            return await original(robot, port_finder)

        factory.build_robot_driver = fail_loading_the_second  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="refused to build"):
            async with _loaded(environment, factory):
                pass

        assert factory.robots["first"].disconnects == 1


@pytest.fixture
def cameras(monkeypatch: pytest.MonkeyPatch) -> dict[str, _FakeSharedCamera]:
    """Replace the SharedCamera builder so no publisher process is spawned."""
    built: dict[str, _FakeSharedCamera] = {}

    def build(row: _CameraRow, **_: Any) -> _FakeSharedCamera:
        return built.setdefault(row.name, _FakeSharedCamera())

    monkeypatch.setattr("utils.camera_factory.build_shared_camera", build)
    return built


def _camera_env() -> tuple[_Environment, _Factory]:
    environment = _Environment(
        robots=[_Configured(robot=_Row("follower"))],
        cameras=[_CameraRow("overhead"), _CameraRow("gripper")],
    )
    return environment, _Factory()


class TestCameras:
    async def test_every_camera_is_attached_and_released(self, cameras: dict[str, _FakeSharedCamera]) -> None:
        environment, factory = _camera_env()

        async with _loaded(environment, factory) as loaded:
            assert set(loaded.state().cameras) == {"overhead", "gripper"}
            assert all(camera.connected for camera in cameras.values())

        assert all(not camera.connected for camera in cameras.values())
        assert all(camera.disconnects == 1 for camera in cameras.values())

    async def test_frames_reach_the_store_on_their_own(self, cameras: dict[str, _FakeSharedCamera]) -> None:
        environment, factory = _camera_env()

        async with _loaded(environment, factory) as loaded:
            await _settle()
            sample = loaded.store.read("observation.images.overhead")

        assert sample is not None
        assert sample.value.shape == (480, 640, 3)
        assert sample.timestamp == 500.0
        assert cameras["overhead"].reads > 0

    async def test_a_publisher_at_another_resolution_is_resized(self, cameras: dict[str, _FakeSharedCamera]) -> None:
        """The environment's resolution leads, so an IP camera serving its own
        size is adapted rather than rejected."""
        environment, factory = _camera_env()
        environment.cameras = [_CameraRow("overhead")]
        cameras["overhead"] = _FakeSharedCamera(shape=(720, 1280, 3))

        async with _loaded(environment, factory) as loaded:
            await _settle()
            sample = loaded.store.read("observation.images.overhead")

        assert sample is not None
        assert sample.value.shape == (480, 640, 3), "the declared resolution did not win"

    async def test_a_camera_that_cannot_attach_releases_the_robots(self, cameras: dict[str, _FakeSharedCamera]) -> None:
        """A busy device is the common failure, and it must not leave an arm energized."""
        environment, factory = _camera_env()
        environment.cameras = [_CameraRow("overhead")]
        cameras["overhead"] = _FakeSharedCamera(connect_error=RuntimeError("device or resource busy"))

        with pytest.raises(RuntimeError, match="busy"):
            async with _loaded(environment, factory):
                pass

        assert all(not robot.connected for robot in factory.robots.values())
        assert all(robot.disconnects == 1 for robot in factory.robots.values())

    async def test_cameras_tick_at_their_own_rate(self, cameras: dict[str, _FakeSharedCamera]) -> None:
        """A 5Hz camera must not be dragged to the robot rate, nor the reverse."""
        environment, factory = _camera_env()
        environment.cameras = [_CameraRow("overhead", fps=5)]

        async with _loaded(environment, factory) as loaded:
            await _settle()

            assert loaded.shape.cameras[0].fps == 5.0
            assert loaded.store.read("observation.images.overhead") is not None


def _write_action(loaded: LoadedEnvironment, value: float) -> None:
    """Write what a control would, without running one."""
    keys = loaded.action_keys_for(loaded.shape.robots[0])
    loaded.store.write_many(dict.fromkeys(keys, value), timestamp=9.0)


class TestFollowing:
    """An arm follows its action features always; there is no switch."""

    async def test_a_freshly_loaded_arm_holds_where_it_is(self) -> None:
        """Loading commands the arm, but only to the position it already has."""
        environment, factory = _teleop_env()

        async with _loaded(environment, factory):
            await _settle()
            resting = factory.robots["follower"].position

        sent = factory.robots["follower"].sent
        assert sent, "the follower was never commanded"
        for command in sent:
            np.testing.assert_allclose(command, [resting] * len(JOINTS))

    async def test_a_written_action_is_followed(self) -> None:
        environment, factory = _teleop_env()

        async with _loaded(environment, factory) as loaded:
            await _settle()
            _write_action(loaded, 4.0)
            await _settle()

        np.testing.assert_allclose(factory.robots["follower"].sent[-1], [4.0, 4.0])

    async def test_the_action_features_start_at_the_measured_position(self) -> None:
        environment, factory = _teleop_env()

        async with _loaded(environment, factory) as loaded:
            resting = factory.robots["follower"].position
            key = loaded.action_keys_for(loaded.shape.robots[0])[0]
            sample = loaded.store.read(key)

        assert sample is not None
        assert sample.value == pytest.approx(resting)

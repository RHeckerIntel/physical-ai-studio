"""The session holds at most one environment, and can swap it."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import pytest

from runtimev2.control.config import ModelControlConfig, TeleopControlConfig
from runtimev2.control.teleop import JointMappingError
from runtimev2.features import ACTION_PREFIX, joint_feature_key
from runtimev2.session import RuntimeSession

if TYPE_CHECKING:
    from pathlib import Path

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
    """Stands in for a connected SharedRobot, including its connect-gated joints."""

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
        # Readable unconnected: the describe step derives a shape that way.
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
        """Called twice per robot: once to describe, once to load."""
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
    return _Environment(robots=[_Configured(robot=_Row("follower"))]), _Factory(roles={"follower": "follower"})


async def _settle() -> None:
    """Give the worker threads a few ticks to run."""
    await asyncio.sleep(0.2)


class TestLoading:
    async def test_a_session_starts_empty(self) -> None:
        """Nothing is connected until an environment is asked for."""
        async with RuntimeSession(_Factory()) as session:
            assert session.environment is None
            assert session.state().loaded is False

    async def test_loading_connects_the_environments_robots(self) -> None:
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            loaded = await session.load(environment)

            assert session.environment is loaded
            assert session.state().loaded is True
            assert all(robot.connected for robot in factory.robots.values())

    async def test_unloading_releases_them_and_leaves_the_session_open(self) -> None:
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)
            await session.unload()

            assert session.environment is None
            assert all(not robot.connected for robot in factory.robots.values())
            # Still usable: the point of unloading is to load something else.
            await session.load(environment)
            assert session.environment is not None

    async def test_unloading_is_idempotent(self) -> None:
        async with RuntimeSession(_Factory()) as session:
            await session.unload()
            await session.unload()

            assert session.environment is None

    async def test_leaving_the_session_unloads_what_is_still_loaded(self) -> None:
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)

        assert all(not robot.connected for robot in factory.robots.values())

    async def test_loading_again_releases_the_previous_robots_first(self) -> None:
        """Two environments can share a robot, so the old one has to let go
        before the new one claims anything."""
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)
            first = dict(factory.robots)
            await session.load(environment)

            assert all(robot.disconnects == 1 for robot in first.values())
            assert all(robot.connected for robot in factory.robots.values())

    async def test_a_failed_load_leaves_the_session_empty(self) -> None:
        environment, factory = _teleop_env()
        factory.joints = {"follower": JOINTS, "leader": ("only_one",)}

        with pytest.raises(JointMappingError, match="cannot drive"):
            async with RuntimeSession(factory) as session:
                await session.load(environment)

        assert all(not robot.connected for robot in factory.robots.values())

    async def test_a_failed_load_does_not_keep_the_previous_environment(self) -> None:
        """The previous one is unloaded first, so a failure cannot silently
        leave the client driving the environment it asked to replace."""
        good, factory = _teleop_env()
        broken = _Environment(
            name="broken", robots=[_Configured(robot=_Row("follower"), tele_operator=_Teleoperator(robot=_Row("odd")))]
        )
        factory.roles = {"follower": "follower", "leader": "leader", "odd": "leader"}

        async with RuntimeSession(factory) as session:
            await session.load(good)
            factory.joints = {"odd": ("only_one",)}

            with pytest.raises(JointMappingError, match="cannot drive"):
                await session.load(broken)

            assert session.environment is None


class TestCommandsNeedAnEnvironment:
    async def test_teleoperating_without_a_loaded_environment_is_refused(self) -> None:
        async with RuntimeSession(_Factory()) as session:
            with pytest.raises(RuntimeError, match="No environment is loaded"):
                session.require_environment()


class _FakeMutation:
    def __init__(self) -> None:
        self.saved = 0
        self.discarded = 0
        self.torn_down = 0

    def add_frame(self, obs: dict, act: dict, task: str) -> None: ...

    def save_episode(self) -> None:
        self.saved += 1

    def discard_buffer(self) -> None:
        self.discarded += 1

    def teardown(self) -> None:
        self.torn_down += 1


@pytest.fixture
def dataset(monkeypatch: pytest.MonkeyPatch) -> list[_FakeMutation]:
    """Open datasets without touching the filesystem."""
    from runtimev2.recording import LoadedDataset

    opened: list[_FakeMutation] = []

    def open_for_recording(dataset_id: Any, path: Any, shape: Any, *, requested_fps: int, **_: Any) -> tuple[Any, Any]:
        mutation = _FakeMutation()
        opened.append(mutation)
        return LoadedDataset(dataset_id=dataset_id, path=path, fps=requested_fps), mutation

    monkeypatch.setattr("runtimev2.session.open_for_recording", open_for_recording)
    return opened


class TestDatasetAboveEnvironment:
    """A dataset is chosen once and outlives any set of connected devices."""

    async def test_a_dataset_survives_an_environment_reload(self, dataset: list[_FakeMutation], tmp_path: Path) -> None:
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)
            await session.load_dataset(uuid4(), tmp_path, hz=10)
            assert session.state().dataset_loaded

            await session.load(environment)

            assert session.state().dataset_loaded, "the swap dropped the dataset"
            assert session.state().dataset_hz == 10
            assert len(dataset) == 2, "it was not reopened against the new shape"

    async def test_a_dataset_can_be_chosen_before_an_environment(
        self, dataset: list[_FakeMutation], tmp_path: Path
    ) -> None:
        """It opens when one arrives, because a row's columns come from the devices."""
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load_dataset(uuid4(), tmp_path, hz=10)
            assert not session.state().dataset_loaded
            assert dataset == []

            await session.load(environment)

            assert session.state().dataset_loaded
            assert len(dataset) == 1

    async def test_unloading_the_environment_keeps_the_dataset(
        self, dataset: list[_FakeMutation], tmp_path: Path
    ) -> None:
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)
            await session.load_dataset(uuid4(), tmp_path, hz=10)
            await session.unload()

            assert session.state().dataset_loaded, "unloading the devices closed the dataset"
            assert dataset[0].torn_down == 0

    async def test_leaving_the_session_closes_the_dataset(self, dataset: list[_FakeMutation], tmp_path: Path) -> None:
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)
            await session.load_dataset(uuid4(), tmp_path, hz=10)

        assert dataset[0].torn_down == 1, "the recording cache was never copied back"


class TestRecordingCommands:
    async def test_recording_needs_a_dataset(self) -> None:
        async with RuntimeSession(_Factory()) as session:
            with pytest.raises(RuntimeError, match="No dataset is open"):
                session.start_recording("a task")

    async def test_an_episode_is_started_and_saved(self, dataset: list[_FakeMutation], tmp_path: Path) -> None:
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)
            await session.load_dataset(uuid4(), tmp_path, hz=10)
            session.start_recording("pick up the cube")
            assert session.state().is_recording
            assert session.state().task == "pick up the cube"

            await session.save_episode()

            assert not session.state().is_recording
            assert session.state().episodes_recorded == 1
            assert dataset[0].saved == 1

    async def test_an_episode_can_be_discarded(self, dataset: list[_FakeMutation], tmp_path: Path) -> None:
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)
            await session.load_dataset(uuid4(), tmp_path, hz=10)
            session.start_recording("a task")

            await session.discard_episode()

            assert dataset[0].discarded == 1
            assert session.state().episodes_recorded == 0


@pytest.fixture
def policy(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Load policies without reading an export or touching an accelerator."""
    from runtimev2.inference import LoadedModel

    loaded: list[Any] = []

    def load_model(model_id: Any, path: Any, shape: Any, *, device: str) -> tuple[Any, Any]:
        tasks: list[Any] = []

        def select_action(observation: dict[str, Any]) -> np.ndarray:
            tasks.append(observation.get("task"))
            return np.zeros(len(JOINTS), dtype=np.float32)

        model = SimpleNamespace(
            reset=lambda: None,
            select_action=select_action,
            adapter=SimpleNamespace(input_names=[]),
            chunk_size=20,
            tasks=tasks,
        )
        loaded.append(model)
        return LoadedModel(model_id=model_id, export_dir=path, chunk_size=20), model

    monkeypatch.setattr("runtimev2.session.load_model", load_model)
    return loaded


class TestPolicyAboveEnvironment:
    async def test_a_policy_survives_an_environment_reload(self, policy: list[Any], tmp_path: Path) -> None:
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)
            await session.load_policy(uuid4(), tmp_path, device="cpu")
            assert session.state().model_loaded

            await session.load(environment)

            assert session.state().model_loaded, "the swap dropped the policy"
            assert len(policy) == 2, "it was not reloaded against the new shape"

    async def test_a_policy_can_be_chosen_before_an_environment(self, policy: list[Any], tmp_path: Path) -> None:
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load_policy(uuid4(), tmp_path, device="cpu")
            assert not session.state().model_loaded

            await session.load(environment)

            assert session.state().model_loaded
            assert len(policy) == 1

    async def test_loading_a_policy_does_not_select_it(self, policy: list[Any], tmp_path: Path) -> None:
        """Selecting a control is what starts motion, so loading must not."""
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)
            resting = factory.robots["follower"].position
            await session.load_policy(uuid4(), tmp_path, device="cpu")
            await _settle()

            assert session.state().control is None
            for command in factory.robots["follower"].sent:
                np.testing.assert_allclose(command, [resting] * len(JOINTS))

    async def test_a_policy_writes_action_features(self, policy: list[Any], tmp_path: Path) -> None:
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)
            await session.load_policy(uuid4(), tmp_path, device="cpu")
            await _settle()
            action = session.require_environment().store.read(joint_feature_key(ACTION_PREFIX, "follower", "gripper"))

        assert action is not None

    async def test_the_task_reaches_a_running_policy(self, policy: list[Any], tmp_path: Path) -> None:
        """One string conditions the policy and labels the recording."""
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)
            await session.load_policy(uuid4(), tmp_path, device="cpu")
            await session.select_control(ModelControlConfig())
            session.set_task("pick up the cube")
            await _settle()

            assert policy[0].tasks, "the policy never inferred"
            assert policy[0].tasks[-1] == ["pick up the cube"]

    async def test_unloading_a_policy_leaves_the_environment(self, policy: list[Any], tmp_path: Path) -> None:
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)
            await session.load_policy(uuid4(), tmp_path, device="cpu")
            await session.unload_policy()

            assert not session.state().model_loaded
            assert session.state().loaded


class TestOnlyOneThingDrives:
    """A human and a policy fighting over the same joints is not a reachable state."""

    async def test_selecting_a_policy_displaces_teleoperation(self, policy: list[Any], tmp_path: Path) -> None:
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)
            await session.select_control(TeleopControlConfig())
            assert session.state().control == TeleopControlConfig()

            await session.load_policy(uuid4(), tmp_path, device="cpu")
            await session.select_control(ModelControlConfig())

            assert isinstance(session.state().control, ModelControlConfig)

    async def test_unloading_a_policy_leaves_nothing_controlling(self, policy: list[Any], tmp_path: Path) -> None:
        """Rather than guessing at teleoperation; the follower holds position."""
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)
            await session.select_control(TeleopControlConfig())
            await session.load_policy(uuid4(), tmp_path, device="cpu")
            await session.select_control(ModelControlConfig())
            await session.unload_policy()

            assert session.state().control is None

    async def test_unloading_a_policy_leaves_another_control_alone(self, policy: list[Any], tmp_path: Path) -> None:
        """Unloading one that was never selected must not clear the control."""
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)
            await session.select_control(TeleopControlConfig())

            await session.unload_policy()

            assert session.state().control == TeleopControlConfig()


class TestImageConversion:
    """Cameras publish (H, W, 3) uint8; vision policies take channels-first float."""

    def test_a_model_declaring_no_preprocessing_is_given_the_conversion(self) -> None:
        """Existing exports declare none, and fail on their first frame without it."""
        from runtimev2.inference import _ensure_image_conversion

        model = SimpleNamespace(preprocessors=[])

        _ensure_image_conversion(model)  # type: ignore[arg-type]

        assert len(model.preprocessors) == 1

    def test_a_models_own_pipeline_is_left_alone(self) -> None:
        """A policy with specific preprocessing would otherwise lose it."""
        from runtimev2.inference import _ensure_image_conversion

        declared = object()
        model = SimpleNamespace(preprocessors=[declared])

        _ensure_image_conversion(model)  # type: ignore[arg-type]

        assert model.preprocessors == [declared]


class TestSelectingAControl:
    """The session chooses; the environment runs what it is given."""

    async def test_a_session_starts_with_nothing_selected(self) -> None:
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)

            assert session.state().control is None

    async def test_teleop_is_selected_by_name(self) -> None:
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)
            await session.select_control(TeleopControlConfig())

            assert session.state().control == TeleopControlConfig()

    async def test_teleop_is_selected_with_its_own_rate(self) -> None:
        """The rate is part of the choice, and comes back in state."""
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)
            await session.select_control(TeleopControlConfig(hz=50.0))

            assert session.state().control == TeleopControlConfig(hz=50.0)

    async def test_teleop_without_a_leader_is_refused(self) -> None:
        environment, factory = _follower_only_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)

            with pytest.raises(RuntimeError, match="no leader"):
                await session.select_control(TeleopControlConfig())

    async def test_a_choice_survives_an_environment_reload(self) -> None:
        """It is rebuilt against the new store, like a dataset worker."""
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)
            await session.select_control(TeleopControlConfig())

            await session.load(environment)

            assert session.state().control == TeleopControlConfig()

    async def test_a_choice_made_before_loading_is_applied_on_load(self) -> None:
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.select_control(TeleopControlConfig())
            assert session.state().environment is None

            await session.load(environment)

            assert session.state().control == TeleopControlConfig()

    async def test_selecting_nothing_releases_the_control(self) -> None:
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)
            await session.select_control(TeleopControlConfig())

            await session.select_control(None)

            assert session.state().control is None
            assert session.state().loaded, "the environment should stay loaded"

    async def test_selecting_a_model_without_one_loaded_is_refused(self) -> None:
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)

            with pytest.raises(RuntimeError, match="No model is loaded"):
                await session.select_control(ModelControlConfig())


class TestSelectingAControlIsEnoughToDrive:
    """There is no second switch: an arm follows its action features always."""

    async def test_selecting_a_control_moves_the_follower(self) -> None:
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)
            factory.robots["leader"].position = 1.5
            await _settle()
            assert session.state().control is None

            await session.select_control(TeleopControlConfig(hz=200.0))
            await _settle()

        sent = factory.robots["follower"].sent
        assert sent, "the follower was never commanded"
        np.testing.assert_allclose(sent[-1], [1.5, 1.5])

    async def test_before_a_control_the_follower_only_holds(self) -> None:
        """It is commanded, but only to where it already was."""
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)
            resting = factory.robots["follower"].position
            factory.robots["leader"].position = 1.5
            await _settle()

        for command in factory.robots["follower"].sent:
            np.testing.assert_allclose(command, [resting] * len(JOINTS))

    async def test_releasing_the_control_leaves_the_arm_where_it_was(self) -> None:
        """Nothing writes the action any more, so the arm holds its last command."""
        environment, factory = _teleop_env()

        async with RuntimeSession(factory) as session:
            await session.load(environment)
            factory.robots["leader"].position = 1.5
            await session.select_control(TeleopControlConfig(hz=200.0))
            await _settle()

            await session.select_control(None)
            await _settle()
            after_release = len(factory.robots["follower"].sent)
            await _settle()

            # Still commanded -- it follows always -- but always the same value.
            assert len(factory.robots["follower"].sent) > after_release
            np.testing.assert_allclose(factory.robots["follower"].sent[-1], [1.5, 1.5])

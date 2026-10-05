"""A policy infers on its own thread; the robot samples the newest result."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from physicalai.inference.constants import IMAGES, STATE, TASK

from runtimev2.control.model import ModelCameraMismatchError, ModelControl, check_camera_keys
from runtimev2.environment import CameraShape, RobotShape, SessionShape
from runtimev2.features import ACTION_PREFIX, OBSERVATION_PREFIX, image_feature_key, joint_feature_key
from runtimev2.store import FeatureStore

JOINTS = ("shoulder_pan", "gripper")


@dataclass
class _Adapter:
    input_names: list[str] = field(default_factory=list)


@dataclass
class _FakeModel:
    """Stands in for an InferenceModel, recording what it was asked."""

    action: np.ndarray = field(default_factory=lambda: np.array([1.0, 2.0], dtype=np.float32))
    adapter: _Adapter = field(default_factory=_Adapter)
    observations: list[dict[str, Any]] = field(default_factory=list)
    resets: int = 0
    delay: float = 0.0

    def reset(self) -> None:
        self.resets += 1

    def select_action(self, observation: dict[str, Any]) -> np.ndarray:
        if self.delay:
            threading.Event().wait(self.delay)
        self.observations.append(observation)
        return self.action


def _shape(cameras: int = 1) -> SessionShape:
    names = ["overhead", "gripper cam"][:cameras]
    return SessionShape(
        robots=(RobotShape(key="follower", robot_id="r0", role="follower", joint_names=JOINTS),),
        cameras=tuple(
            CameraShape(key=name.replace(" ", "_"), name=name, camera_id=f"c{i}", shape=(480, 640, 3), fps=30.0)
            for i, name in enumerate(names)
        ),
    )


def _action_key(joint: str) -> str:
    return joint_feature_key(ACTION_PREFIX, "follower", joint)


def _filled(shape: SessionShape) -> FeatureStore:
    store = FeatureStore(shape.feature_spec())
    for index, joint in enumerate(JOINTS):
        store.write(joint_feature_key(OBSERVATION_PREFIX, "follower", joint), float(index), timestamp=7.0)
    for camera in shape.cameras:
        store.write(image_feature_key(camera.key), np.zeros(camera.shape, np.uint8), timestamp=7.0)
    return store


class TestWhatItCommands:
    def test_a_tick_writes_the_action_features(self) -> None:
        shape = _shape()
        store = _filled(shape)
        control = ModelControl(_FakeModel(), store, shape, hz=10.0)

        control.tick()

        assert store.read(_action_key("shoulder_pan")).value == 1.0
        assert store.read(_action_key("gripper")).value == 2.0

    def test_the_action_carries_the_observations_time(self) -> None:
        """The gap between them is the period the robot interpolates across."""
        shape = _shape()
        store = _filled(shape)

        ModelControl(_FakeModel(), store, shape, hz=10.0).tick()

        assert store.read(_action_key("gripper")).timestamp == 7.0

    def test_nothing_is_written_before_the_first_inference(self) -> None:
        shape = _shape()
        store = _filled(shape)

        ModelControl(_FakeModel(), store, shape, hz=10.0)

        assert store.read(_action_key("gripper")) is None

    def test_an_incomplete_observation_is_skipped(self) -> None:
        """A policy fed a default where an image belongs produces a confident
        action from data that was never measured."""
        shape = _shape()
        store = FeatureStore(shape.feature_spec())
        store.write(joint_feature_key(OBSERVATION_PREFIX, "follower", "gripper"), 1.0, timestamp=1.0)
        model = _FakeModel()

        ModelControl(model, store, shape, hz=10.0).tick()

        assert model.observations == []
        assert store.read(_action_key("gripper")) is None

    def test_it_is_named_for_a_client_to_show(self) -> None:
        shape = _shape()

        assert ModelControl(_FakeModel(), _filled(shape), shape, hz=10.0).name == "model"


class TestModelInput:
    def test_state_is_batched_in_joint_order(self) -> None:
        shape = _shape()
        model = _FakeModel()

        ModelControl(model, _filled(shape), shape, hz=10.0).tick()

        np.testing.assert_allclose(model.observations[0][STATE], [[0.0, 1.0]])

    def test_one_camera_uses_the_bare_images_key(self) -> None:
        """A single-camera model discards the name."""
        shape = _shape(cameras=1)
        model = _FakeModel()

        ModelControl(model, _filled(shape), shape, hz=10.0).tick()

        assert model.observations[0][IMAGES].shape == (1, 480, 640, 3)

    def test_several_cameras_are_named_as_the_dataset_names_them(self) -> None:
        """The model was trained against the dataset's camera keys, which keep spaces."""
        shape = _shape(cameras=2)
        model = _FakeModel()

        ModelControl(model, _filled(shape), shape, hz=10.0).tick()

        assert set(model.observations[0]) >= {f"{IMAGES}.overhead", f"{IMAGES}.gripper cam"}

    def test_a_task_is_passed_when_set(self) -> None:
        shape = _shape()
        model = _FakeModel()

        ModelControl(model, _filled(shape), shape, hz=10.0, task="pick up the cube").tick()

        assert model.observations[0][TASK] == ["pick up the cube"]

    def test_no_task_means_no_task_key(self) -> None:
        shape = _shape()
        model = _FakeModel()

        ModelControl(model, _filled(shape), shape, hz=10.0).tick()

        assert TASK not in model.observations[0]


class TestItInfersOffTheRobotsThread:
    async def test_entering_runs_inference_on_its_own_thread(self) -> None:
        shape = _shape()
        model = _FakeModel()
        control = ModelControl(model, _filled(shape), shape, hz=200.0)

        async with control:
            await asyncio.sleep(0.05)

        assert model.resets == 1, "a stale chunk would act on observations no longer true"
        assert len(model.observations) > 1, "its own thread never inferred"

    async def test_slow_inference_does_not_hold_the_store(self) -> None:
        """A robot reading action features must never wait on inference."""
        shape = _shape()
        store = _filled(shape)
        control = ModelControl(_FakeModel(delay=0.2), store, shape, hz=100.0)

        async with control:
            await asyncio.sleep(0.02)
            start = asyncio.get_running_loop().time()
            for _ in range(50):
                store.snapshot((_action_key("gripper"),))
            elapsed = asyncio.get_running_loop().time() - start

        assert elapsed < 0.05, f"reading the store blocked for {elapsed:.3f}s"


class TestCameraCheck:
    def test_a_model_naming_a_missing_camera_is_refused(self) -> None:
        model = _FakeModel(adapter=_Adapter(input_names=[STATE, f"{IMAGES}.wrist"]))

        with pytest.raises(ModelCameraMismatchError, match="wrist"):
            check_camera_keys(model, ["overhead"])

    def test_a_single_camera_model_is_not_checked(self) -> None:
        check_camera_keys(_FakeModel(adapter=_Adapter(input_names=[STATE, IMAGES])), ["anything"])

    def test_a_model_with_no_images_is_not_checked(self) -> None:
        check_camera_keys(_FakeModel(adapter=_Adapter(input_names=[STATE])), [])

    def test_a_matching_model_passes(self) -> None:
        check_camera_keys(_FakeModel(adapter=_Adapter(input_names=[STATE, f"{IMAGES}.overhead"])), ["overhead"])

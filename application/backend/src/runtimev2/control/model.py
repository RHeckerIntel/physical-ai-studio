"""Drive a robot from a policy, inferring on a thread of its own.

The one control that cannot be a plain function: inference takes ten ticks of a
100Hz arm, so it cannot run where the robot samples it. It runs on this worker's
thread instead, and :meth:`action` hands back the newest result.

``select_action`` keeps the chunk it predicted and returns one action per call,
so this worker's ``hz`` is the rate actions are *emitted* -- the rate the robot
is commanded at -- while real inference happens every ``chunk_size`` ticks.
Interpolating between actions would make the motion smoother; that is separate.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger
from physicalai.inference.constants import IMAGES, STATE, TASK

from runtime.features import sanitize_camera_name
from runtimev2.control.base import ControlAction, ControlAlgorithm
from runtimev2.features import ACTION_PREFIX, OBSERVATION_PREFIX, image_feature_key, joint_feature_key

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicalai.inference import InferenceModel

    from runtimev2.environment import SessionShape
    from runtimev2.session_store import SessionStore


class ModelCameraMismatchError(RuntimeError):
    """Raised when a model names image inputs this session does not have."""

    def __init__(self, expected: list[str], provided: list[str]) -> None:
        super().__init__(f"Model expects images {expected}, but this environment provides {provided}")


def check_camera_keys(model: InferenceModel, camera_keys: list[str]) -> None:
    """Reject a model whose named image inputs are not among these cameras.

    Single-camera models emit a bare ``images`` key and discard the name, so
    they are not checked, nor are models declaring no image inputs.

    Raises:
        ModelCameraMismatchError: A named input has no camera behind it.
    """
    expected = {name for name in model.adapter.input_names if name == IMAGES or name.startswith(f"{IMAGES}.")}
    if not expected or expected == {IMAGES}:
        return
    provided = {f"{IMAGES}.{key}" for key in camera_keys}
    missing = expected - provided
    if missing:
        raise ModelCameraMismatchError(sorted(expected), sorted(provided))


class ModelControl(ControlAlgorithm):
    """Infer from the store's observations; hand the newest action to the robot.

    Holds no device. The model is loaded above the environment and outlives this.
    """

    def __init__(
        self,
        model: InferenceModel,
        store: SessionStore,
        shape: SessionShape,
        *,
        hz: float,
        task: str | None = None,
    ) -> None:
        follower = shape.robots[0]
        super().__init__(
            store,
            [joint_feature_key(ACTION_PREFIX, follower.key, joint) for joint in follower.joint_names],
            name="model",
            hz=hz,
        )
        self._model = model
        self._task = task
        # The order the model's state vector was trained in, which is the
        # shape's joint order.
        self._state_keys = [joint_feature_key(OBSERVATION_PREFIX, follower.key, j) for j in follower.joint_names]
        # Image inputs use the dataset's camera naming, because that is what the
        # model was trained against.
        self._images = {image_feature_key(camera.key): sanitize_camera_name(camera.name) for camera in shape.cameras}
        self._skipped = 0

    @property
    def task(self) -> str | None:
        return self._task

    @task.setter
    def task(self, task: str | None) -> None:
        self._task = task

    @contextmanager
    def acquire(self) -> Generator[None]:
        """Reset the policy, and report what it had to skip.

        A policy carries history, so a chunk left from a previous attachment
        would act on observations that are no longer true.
        """
        self._model.reset()
        try:
            yield
        finally:
            if self._skipped:
                logger.info("Inference skipped {} ticks for want of a complete observation", self._skipped)

    def compute(self) -> ControlAction | None:
        """Infer one action from the store's observations.

        An incomplete observation is skipped, not padded: a policy fed a default
        where a measurement belongs answers confidently from data it never saw.
        """
        wanted = (*self._state_keys, *self._images)
        snapshot = self._store.snapshot(wanted)
        if len(snapshot) != len(wanted):
            self._skipped += 1
            return None
        values = self._model.select_action(self._observation(snapshot))
        return ControlAction(
            values=np.asarray(values, dtype=np.float32),
            # The observation's time, not now, so a reader can see how old the
            # measurement behind a command was.
            timestamp=max(snapshot[key].timestamp for key in self._state_keys),
        )

    def _observation(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Assemble the model's input in the shape it was exported with."""
        state = np.array([[snapshot[key].value for key in self._state_keys]], dtype=np.float32)
        observation: dict[str, Any] = {STATE: state}
        images = {name: snapshot[key].value[np.newaxis] for key, name in self._images.items()}
        if len(images) > 1:
            for name, data in images.items():
                observation[f"{IMAGES}.{name}"] = data
        elif images:
            # A single-camera model takes a bare key and discards the name.
            observation[IMAGES] = next(iter(images.values()))
        if self._task is not None:
            observation[TASK] = [self._task]
        return observation

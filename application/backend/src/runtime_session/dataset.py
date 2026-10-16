"""What a dataset needs from a session, read from the dataset on disk.

A dataset stores the follower's joints packed into ``observation.state`` and
``action`` with ``<joint>.pos`` names, and each camera as
``observation.images.<key>``. The store uses the same names and keys, so a
dataset's spec compares to a session's directly. A dataset with nothing
recorded yet has no shape of its own and takes the session's.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from runtime_session.store import FeatureSpec, ImageSpec

if TYPE_CHECKING:
    from schemas.dataset import Dataset

DEFAULT_RECORDING_FPS = 30
"""The rate a new dataset is recorded at; an existing one keeps its own."""

_IMAGE_PREFIX = "observation.images."


def _info(dataset: Dataset) -> dict[str, Any] | None:
    info = Path(dataset.path) / "meta" / "info.json"
    return json.loads(info.read_text()) if info.is_file() else None


def dataset_fps(dataset: Dataset) -> int:
    """The rate the dataset was recorded at, or the default for a new one."""
    info = _info(dataset)
    return int(info["fps"]) if info is not None else DEFAULT_RECORDING_FPS


def dataset_spec(dataset: Dataset) -> FeatureSpec | None:
    """The joints and cameras the dataset was recorded with, or ``None`` when nothing is recorded yet.

    Raises:
        ValueError: The dataset's state and action name different joints.
    """
    info = _info(dataset)
    if info is None:
        return None
    features: dict[str, Any] = info["features"]
    joints = tuple(features["observation.state"]["names"])
    if tuple(features["action"]["names"]) != joints:
        raise ValueError(f"Dataset {dataset.name!r} records an action for other joints than its state")
    cameras = {
        key.removeprefix(_IMAGE_PREFIX): ImageSpec(*(int(dim) for dim in value["shape"]))
        for key, value in features.items()
        if key.startswith(_IMAGE_PREFIX)
    }
    return FeatureSpec(joints, cameras)


def lerobot_features(spec: FeatureSpec) -> dict[str, Any]:
    """The LeRobot feature metadata a new dataset is created with, from the session's spec."""
    # Imported here: lerobot is heavy, and only the dataset worker's process needs it.
    from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
    from lerobot.processor import make_default_processors
    from lerobot.utils.feature_utils import combine_feature_dicts

    teleop_action_processor, _robot_action_processor, robot_observation_processor = make_default_processors()
    action: dict[str, Any] = dict.fromkeys(spec.joint_names, float)
    observation: dict[str, Any] = dict.fromkeys(spec.joint_names, float)
    observation.update({key: image.shape for key, image in spec.observations.items()})
    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=action),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=observation),
            use_videos=True,
        ),
    )

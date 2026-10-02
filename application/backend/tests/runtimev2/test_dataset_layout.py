"""Compatibility is judged on the packed layout a dataset and a model use."""

from __future__ import annotations

import pytest

from runtimev2.dataset_layout import ACTION_KEY, STATE_KEY, DatasetLayout, LayoutEntry
from runtimev2.environment import CameraShape, EnvironmentShape, RobotShape

JOINTS = ("shoulder_pan", "elbow_flex", "gripper")
PACKED = ("shoulder_pan.pos", "elbow_flex.pos", "gripper.pos")


def _shape(*, followers: int = 1, leader: bool = False, cameras: bool = True) -> EnvironmentShape:
    robots = [
        RobotShape(key=f"follower{index or ''}", robot_id=f"r{index}", role="follower", joint_names=JOINTS)
        for index in range(followers)
    ]
    if leader:
        robots.append(RobotShape(key="leader", robot_id="rl", role="leader", joint_names=JOINTS))
    camera_shapes = (
        [CameraShape(key="overhead", name="overhead", camera_id="c0", shape=(480, 640, 3), fps=30.0)] if cameras else []
    )
    return EnvironmentShape(robots=tuple(robots), cameras=tuple(camera_shapes))


class TestProjection:
    def test_joints_are_packed_with_unprefixed_names(self) -> None:
        """A robot's session key must not reach disk, or renaming it would break datasets."""
        layout = DatasetLayout.from_environment(_shape())

        assert layout.entries[ACTION_KEY] == LayoutEntry(shape=(3,), names=PACKED)
        assert layout.entries[STATE_KEY] == LayoutEntry(shape=(3,), names=PACKED)

    def test_cameras_become_image_entries(self) -> None:
        layout = DatasetLayout.from_environment(_shape())

        assert layout.entries["observation.images.overhead"] == LayoutEntry(shape=(480, 640, 3))

    def test_a_leader_is_not_recorded(self) -> None:
        """A leader is a control source, not something a dataset stores."""
        assert DatasetLayout.from_environment(_shape(leader=True)).entries[ACTION_KEY].names == PACKED

    def test_several_followers_have_no_defined_order(self) -> None:
        """Nothing in the environment says whose joints come first; guessing would
        produce datasets that are silently incompatible with each other."""
        with pytest.raises(ValueError, match="exactly one follower"):
            DatasetLayout.from_environment(_shape(followers=2))

    def test_no_follower_is_refused(self) -> None:
        with pytest.raises(ValueError, match="exactly one follower"):
            DatasetLayout.from_environment(_shape(followers=0))


class TestFromInfo:
    def _info(self) -> dict[str, object]:
        return {
            "action": {"shape": [3], "names": list(PACKED)},
            "observation.state": {"shape": [3], "names": list(PACKED)},
            "observation.images.overhead": {"shape": [480, 640, 3], "names": ["height", "width", "channels"]},
            "timestamp": {"shape": [1], "names": None},
            "frame_index": {"shape": [1], "names": None},
            "index": {"shape": [1], "names": None},
        }

    def test_bookkeeping_columns_are_ignored(self) -> None:
        """They say nothing about whether an environment can produce this dataset."""
        layout = DatasetLayout.from_info(self._info())

        assert set(layout.entries) == {ACTION_KEY, STATE_KEY, "observation.images.overhead"}

    def test_a_real_dataset_matches_the_environment_that_made_it(self) -> None:
        recorded = DatasetLayout.from_info(self._info())

        assert DatasetLayout.from_environment(_shape()).satisfies(recorded)


class TestCompatibility:
    def test_an_extra_camera_is_allowed(self) -> None:
        """An environment with a spare camera can still run a model trained without it."""
        required = DatasetLayout.from_environment(_shape(cameras=False))
        richer = DatasetLayout.from_environment(_shape())

        assert richer.satisfies(required)

    def test_a_missing_camera_is_named(self) -> None:
        required = DatasetLayout.from_environment(_shape())
        poorer = DatasetLayout.from_environment(_shape(cameras=False))

        assert poorer.incompatibilities(required) == ["observation.images.overhead is missing"]

    def test_a_resolution_mismatch_reports_both_shapes(self) -> None:
        required = DatasetLayout.from_info(
            {"observation.images.overhead": {"shape": [720, 1280, 3], "names": ["height", "width", "channels"]}}
        )

        reasons = DatasetLayout.from_environment(_shape()).incompatibilities(required)

        assert reasons == ["observation.images.overhead has shape (480, 640, 3), expected (720, 1280, 3)"]

    def test_a_different_joint_set_reports_the_components(self) -> None:
        """Same joint count, different joints -- the shapes match and only the names catch it."""
        required = DatasetLayout.from_info(
            {"action": {"shape": [3], "names": ["shoulder_pan.pos", "wrist_roll.pos", "gripper.pos"]}}
        )

        reasons = DatasetLayout.from_environment(_shape()).incompatibilities(required)

        assert reasons == [
            "action has components (shoulder_pan.pos, elbow_flex.pos, gripper.pos), "
            "expected (shoulder_pan.pos, wrist_roll.pos, gripper.pos)"
        ]

    def test_a_different_joint_count_reports_the_shape(self) -> None:
        required = DatasetLayout.from_info({"action": {"shape": [6], "names": [f"j{index}.pos" for index in range(6)]}})

        reasons = DatasetLayout.from_environment(_shape()).incompatibilities(required)

        assert "action has shape (3,), expected (6,)" in reasons

"""The feature spec is the session's contract with datasets and models."""

from __future__ import annotations

import pytest

from runtimev2.features import (
    FeatureSpec,
    camera_features,
    image_feature_key,
    joint_feature_key,
    robot_features,
    sanitize_name,
)

JOINTS = ["shoulder_pan", "elbow_flex", "gripper"]


def _spec() -> FeatureSpec:
    return FeatureSpec.build(
        robot_features("follower", JOINTS),
        camera_features({"overhead": (480, 640, 3), "gripper": (480, 640, 3)}),
    )


class TestShape:
    def test_a_robot_gets_both_kinds_for_every_joint(self) -> None:
        """A robot that is only read still declares actions, so driving it later is not a shape change."""
        spec = FeatureSpec.build(robot_features("follower", JOINTS))

        assert spec.keys("observation") == tuple(
            sorted(joint_feature_key("observation", "follower", joint) for joint in JOINTS)
        )
        assert spec.keys("action") == tuple(sorted(joint_feature_key("action", "follower", joint) for joint in JOINTS))

    def test_cameras_are_observations_with_a_shape(self) -> None:
        spec = FeatureSpec.build(camera_features({"overhead": (480, 640, 3)}))
        feature = spec[image_feature_key("overhead")]

        assert feature.kind == "observation"
        assert feature.shape == (480, 640, 3)
        assert feature.dtype == "uint8"
        assert feature.is_image

    def test_two_robots_do_not_collide(self) -> None:
        spec = FeatureSpec.build(robot_features("leader", JOINTS), robot_features("follower", JOINTS))

        assert len(spec.keys()) == 4 * len(JOINTS)

    def test_the_order_is_reproducible(self) -> None:
        """A dataset's columns are matched by name, but a stable order keeps diffs readable."""
        first = FeatureSpec.build(robot_features("follower", JOINTS))
        second = FeatureSpec.build(robot_features("follower", list(reversed(JOINTS))))

        assert first.keys() == second.keys()

    def test_duplicate_keys_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate feature key"):
            FeatureSpec.build(robot_features("follower", JOINTS), robot_features("follower", JOINTS))


class TestSanitize:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("Overhead Cam", "overhead_cam"),
            ("  Follower Arm #2 ", "follower_arm_2"),
            ("already-safe_1", "already-safe_1"),
            ("Ärm", "_rm"),
        ],
    )
    def test_a_display_name_becomes_a_key(self, name: str, expected: str) -> None:
        """A key has to survive being a dict key, a zenoh key expression and a column."""
        assert sanitize_name(name) == expected

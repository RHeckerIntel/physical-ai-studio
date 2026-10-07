"""The feature spec is the session's contract with datasets and models."""

from __future__ import annotations

import pytest

from runtimev2.features import (
    ACTION_KEY,
    STATE_KEY,
    FeatureSpec,
    camera_features,
    image_feature_key,
    leader_features,
    robot_features,
    sanitize_name,
)

JOINTS = ["shoulder_pan", "elbow_flex", "gripper"]


def _spec() -> FeatureSpec:
    return FeatureSpec.build(
        robot_features(JOINTS),
        camera_features({"overhead": (480, 640, 3), "gripper": (480, 640, 3)}),
    )


class TestShape:
    def test_the_driven_robot_uses_the_well_known_keys(self) -> None:
        """Not one feature per joint, and not keyed by robot: a session drives
        one, so a control can assume both keys exist."""
        spec = FeatureSpec.build(robot_features(JOINTS))

        assert spec.keys("observation") == (STATE_KEY,)
        assert spec.keys("action") == (ACTION_KEY,)

    def test_a_vector_carries_its_joint_names_in_driver_order(self) -> None:
        """Order is the thing a dataset column and ``send_action`` both rely on."""
        spec = FeatureSpec.build(robot_features(JOINTS))
        feature = spec[STATE_KEY]

        assert feature.names == tuple(JOINTS)
        assert feature.shape == (len(JOINTS),)
        assert not feature.is_image

    def test_cameras_are_observations_with_a_shape(self) -> None:
        spec = FeatureSpec.build(camera_features({"overhead": (480, 640, 3)}))
        feature = spec[image_feature_key("overhead")]

        assert feature.kind == "observation"
        assert feature.shape == (480, 640, 3)
        assert feature.dtype == "uint8"
        assert feature.is_image

    def test_a_leader_does_not_collide_with_the_driven_robot(self) -> None:
        """A leader is keyed by name; the driven robot has the singular keys."""
        spec = FeatureSpec.build(robot_features(JOINTS), leader_features("leader", JOINTS))

        assert set(spec.keys()) == {STATE_KEY, ACTION_KEY, "observation.leader"}

    def test_the_order_is_reproducible(self) -> None:
        """A dataset's columns are matched by name, but a stable order keeps diffs readable."""
        first = FeatureSpec.build(robot_features(JOINTS))
        second = FeatureSpec.build(robot_features(list(reversed(JOINTS))))

        assert first.keys() == second.keys()

    def test_duplicate_keys_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate feature key"):
            FeatureSpec.build(robot_features(JOINTS), robot_features(JOINTS))


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

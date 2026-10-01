"""The feature spec is the session's contract with datasets and models."""

from __future__ import annotations

import pytest

from runtimev2.features import (
    Feature,
    FeatureSpec,
    camera_features,
    image_feature_key,
    joint_feature_key,
    robot_features,
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


class TestCompatibility:
    def test_an_identical_spec_is_satisfied(self) -> None:
        assert _spec().satisfies(_spec())

    def test_extra_features_are_allowed(self) -> None:
        """An environment with a second camera can still run a model trained without it."""
        required = FeatureSpec.build(robot_features("follower", JOINTS))

        assert _spec().satisfies(required)

    def test_a_missing_feature_is_named(self) -> None:
        required = FeatureSpec.build(robot_features("follower", [*JOINTS, "wrist_roll"]))

        reasons = _spec().missing_from(required)

        assert reasons == [
            "action.follower.wrist_roll.pos is missing",
            "observation.follower.wrist_roll.pos is missing",
        ]

    def test_a_resolution_mismatch_reports_both_shapes(self) -> None:
        """The actionable fix is reselecting the camera, so the message has to say what it found."""
        required = FeatureSpec.build(camera_features({"overhead": (720, 1280, 3)}))

        reasons = _spec().missing_from(required)

        assert reasons == ["observation.images.overhead has shape (480, 640, 3), expected (720, 1280, 3)"]
        assert not _spec().satisfies(required)

    def test_a_kind_mismatch_is_reported(self) -> None:
        required = FeatureSpec((Feature(joint_feature_key("observation", "follower", "gripper"), "action"),))

        reasons = _spec().missing_from(required)

        assert reasons == ["observation.follower.gripper.pos is an observation, expected an action"]

    def test_every_problem_is_reported_not_just_the_first(self) -> None:
        required = FeatureSpec.build(
            robot_features("follower", ["nonexistent"]),
            camera_features({"overhead": (720, 1280, 3)}),
        )

        assert len(_spec().missing_from(required)) == 3

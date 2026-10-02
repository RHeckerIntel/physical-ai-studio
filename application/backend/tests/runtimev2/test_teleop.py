"""Teleoperation is a key mapping over the store, not a mode."""

from __future__ import annotations

import pytest

from runtimev2.features import ACTION_PREFIX, OBSERVATION_PREFIX, FeatureSpec, joint_feature_key, robot_features
from runtimev2.store import FeatureStore
from runtimev2.workers.teleop import TeleopSource, joint_mapping

JOINTS = ["shoulder_pan", "gripper"]


def _store() -> FeatureStore:
    return FeatureStore(FeatureSpec.build(robot_features("leader", JOINTS), robot_features("follower", JOINTS)))


def _obs(robot: str, joint: str) -> str:
    return joint_feature_key(OBSERVATION_PREFIX, robot, joint)


def _act(robot: str, joint: str) -> str:
    return joint_feature_key(ACTION_PREFIX, robot, joint)


def _source(store: FeatureStore) -> TeleopSource:
    return TeleopSource(
        store,
        joint_mapping(
            tuple(_obs("leader", joint) for joint in JOINTS),
            tuple(_act("follower", joint) for joint in JOINTS),
        ),
    )


class TestForwarding:
    def test_a_leaders_observation_becomes_the_followers_action(self) -> None:
        store = _store()
        store.write_many({_obs("leader", joint): 1.5 for joint in JOINTS}, timestamp=10.0)

        _source(store).tick()

        assert store.read(_act("follower", "gripper")).value == 1.5

    def test_the_action_keeps_the_observations_timestamp(self) -> None:
        """A reader has to be able to tell how old the command it is acting on is."""
        store = _store()
        store.write(_obs("leader", "gripper"), 1.0, timestamp=42.0)

        _source(store).tick()

        assert store.read(_act("follower", "gripper")).timestamp == 42.0

    def test_an_unwritten_leader_joint_is_skipped(self) -> None:
        """Writing a placeholder would only make the follower move sooner on worse data."""
        store = _store()
        store.write(_obs("leader", "gripper"), 1.0, timestamp=1.0)

        _source(store).tick()

        assert store.read(_act("follower", "gripper")) is not None
        assert store.read(_act("follower", "shoulder_pan")) is None

    def test_it_reports_what_it_authors(self) -> None:
        """A session needs to know which keys have a source before enabling writes."""
        assert set(_source(_store()).targets) == {_act("follower", joint) for joint in JOINTS}

    def test_it_has_a_name_for_the_rate_loop(self) -> None:
        assert _source(_store()).name == "teleop"


class TestMappingValidation:
    def test_keys_outside_the_spec_are_refused(self) -> None:
        """A typo would otherwise forward nothing, looking like a dead leader."""
        with pytest.raises(ValueError, match="not a feature of this session"):
            TeleopSource(_store(), {"observation.nope.pos": _act("follower", "gripper")})

    def test_mismatched_joint_counts_are_refused(self) -> None:
        """There is no defensible pairing, so guessing one would move the wrong joints."""
        with pytest.raises(ValueError, match="same number of joints"):
            joint_mapping(("a", "b", "c"), ("x", "y"))

    def test_the_pairing_is_positional(self) -> None:
        """A leader and follower are often different types; joint i drives joint i."""
        assert joint_mapping(("l0", "l1"), ("f0", "f1")) == {"l0": "f0", "l1": "f1"}


class TestPartialAuthorship:
    def test_a_second_source_can_own_other_joints(self) -> None:
        """A keyboard driving one joint and a leader the rest need not know each other."""
        store = _store()
        leader_drives = TeleopSource(store, {_obs("leader", "shoulder_pan"): _act("follower", "shoulder_pan")})
        store.write(_obs("leader", "shoulder_pan"), 3.0, timestamp=1.0)
        store.write(_act("follower", "gripper"), 9.0, timestamp=1.0)  # the "keyboard"

        leader_drives.tick()

        assert store.read(_act("follower", "shoulder_pan")).value == 3.0
        assert store.read(_act("follower", "gripper")).value == 9.0

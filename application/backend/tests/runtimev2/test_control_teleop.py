"""Teleop reads the leader device itself, so nothing sits between the two."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from runtimev2.control.teleop import JointMappingError, TeleopControl, check_pairing
from runtimev2.environment import LeaderShape, RobotShape, SessionShape
from runtimev2.features import ACTION_PREFIX, OBSERVATION_PREFIX, joint_feature_key
from runtimev2.store import FeatureStore

JOINTS = ("shoulder_pan", "gripper")
ACTION_KEYS = tuple(joint_feature_key(ACTION_PREFIX, "follower", joint) for joint in JOINTS)


def _control(leader: _FakeLeader, store: FeatureStore) -> TeleopControl:
    return TeleopControl(leader, store, ACTION_KEYS, hz=100.0)


class _FakeLeader:
    """Stands in for a LeaderDevice: hands back a position and its capture time."""

    def __init__(self, joints: tuple[str, ...] = JOINTS, *, key: str = "leader") -> None:
        self.key = key
        self.joint_names = joints
        self.positions = np.array([0.0, 1.0], dtype=np.float32)
        self.timestamp = 42.0
        self.reads = 0

    def read(self) -> tuple[np.ndarray, float]:
        self.reads += 1
        return self.positions, self.timestamp


def _store() -> FeatureStore:
    shape = SessionShape(
        robots=(RobotShape(key="follower", robot_id="r1", role="follower", joint_names=JOINTS),),
        cameras=(),
        leaders=(LeaderShape(key="leader", robot_id="r0", joint_names=JOINTS),),
    )
    return FeatureStore(shape.feature_spec())


class TestWhatItCommands:
    def test_it_offers_the_leaders_position(self) -> None:
        control = _control(_FakeLeader(), _store())

        action = control.compute()

        np.testing.assert_allclose(action.values, [0.0, 1.0])

    def test_it_carries_the_leaders_capture_time(self) -> None:
        """Which is what makes the follower command once per leader reading."""
        control = _control(_FakeLeader(), _store())

        assert control.compute().timestamp == 42.0

    def test_it_reads_the_device_rather_than_the_store(self) -> None:
        """No clock between reading the leader and producing the command."""
        leader = _FakeLeader()
        control = _control(leader, _store())

        control.compute()
        control.compute()

        assert leader.reads == 2

    def test_it_is_named_for_a_client_to_show(self) -> None:
        assert _control(_FakeLeader(), _store()).name == "teleop"


class TestWhatItPublishes:
    def test_the_leaders_position_reaches_the_store(self) -> None:
        store = _store()
        control = _control(_FakeLeader(), store)

        control.compute()

        assert store.read(joint_feature_key(OBSERVATION_PREFIX, "leader", "gripper")).value == 1.0

    def test_a_tick_writes_the_action_features(self) -> None:
        """A control writes; the robot reads. Nothing hands the two together."""
        store = _store()
        control = _control(_FakeLeader(), store)

        control.tick()

        assert store.read(ACTION_KEYS[1]).value == 1.0
        assert store.read(ACTION_KEYS[1]).timestamp == 42.0

    def test_the_published_position_is_the_one_commanded(self) -> None:
        """One read, two uses, so the record and the command cannot disagree."""
        store = _store()
        leader = _FakeLeader()
        control = _control(leader, store)

        action = control.compute()
        published = store.read(joint_feature_key(OBSERVATION_PREFIX, "leader", "gripper"))

        assert published.value == pytest.approx(action.values[1])
        assert published.timestamp == action.timestamp


class TestPairing:
    def test_mismatched_counts_are_refused(self) -> None:
        with pytest.raises(JointMappingError, match="3 joints cannot drive"):
            check_pairing(["a", "b", "c"], ["x", "y"])

    def test_equal_counts_pass(self) -> None:
        check_pairing(["a", "b"], ["x", "y"])

    def test_a_control_refuses_a_mismatched_pair_when_built(self) -> None:
        """On load, rather than part-way through commanding an arm."""
        with pytest.raises(JointMappingError):
            TeleopControl(_FakeLeader(), _store(), ("only_one",), hz=100.0)


class TestLifecycle:
    async def test_it_writes_from_its_own_thread(self) -> None:
        """Its rate is the leader's, independent of the robot it drives."""
        store = _store()
        control = TeleopControl(_FakeLeader(), store, ACTION_KEYS, hz=200.0)

        async with control:
            await asyncio.sleep(0.05)

        assert store.read(ACTION_KEYS[0]) is not None, "its own thread never wrote"

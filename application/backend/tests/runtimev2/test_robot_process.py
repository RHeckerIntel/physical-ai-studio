"""A robot's loop in its own process, reached only through shared memory."""

from __future__ import annotations

import asyncio

import pytest

from runtimev2.environment import RobotShape, SessionShape
from runtimev2.features import ACTION_PREFIX, OBSERVATION_PREFIX, joint_feature_key
from runtimev2.robot_process import RobotProcess, RobotRecipe
from runtimev2.session_store import SessionStore

JOINTS = ("shoulder_pan", "gripper")
SHAPE = RobotShape(key="arm", robot_id="r0", role="follower", joint_names=JOINTS)
SESSION = SessionShape(robots=(SHAPE,), cameras=())

# A driver the child can rebuild from the recipe alone, with no hardware.
FAKE_DRIVER = {
    "class_path": "tests.runtimev2.fake_driver.FakeDriver",
    "init_args": {"joint_names": list(JOINTS), "position": 1.5},
}


@pytest.fixture
def store() -> SessionStore:
    created = SessionStore.create(SESSION.feature_spec())
    try:
        yield created
    finally:
        created.close()


def _recipe(**overrides: object) -> RobotRecipe:
    return RobotRecipe(shape=SHAPE, driver={**FAKE_DRIVER, **overrides}, hz=200.0)


class TestTheProcess:
    async def test_it_publishes_observations_the_parent_can_read(self, store: SessionStore) -> None:
        """The whole point: the loop runs elsewhere, the truth is shared."""
        key = joint_feature_key(OBSERVATION_PREFIX, "arm", "gripper")

        async with RobotProcess(_recipe(), SESSION, store.shared_name):
            await asyncio.sleep(0.4)
            sample = store.read(key)

        assert sample is not None, "the child never published"
        assert sample.value == pytest.approx(1.5)

    async def test_it_follows_actions_the_parent_writes(self, store: SessionStore) -> None:
        async with RobotProcess(_recipe(), SESSION, store.shared_name):
            await asyncio.sleep(0.3)
            store.write_many(
                {joint_feature_key(ACTION_PREFIX, "arm", joint): 4.0 for joint in JOINTS},
                timestamp=1_000.0,
            )
            await asyncio.sleep(0.4)
            commanded = store.read(joint_feature_key(ACTION_PREFIX, "arm", "gripper"))

        assert commanded is not None
        assert commanded.value == pytest.approx(4.0)

    async def test_a_driver_that_will_not_build_is_reported(self, store: SessionStore) -> None:
        """The child cannot raise into the parent, so the reason travels back."""
        broken = RobotRecipe(shape=SHAPE, driver={"class_path": "nope.NotAThing", "init_args": {}}, hz=50.0)

        with pytest.raises(RuntimeError, match="failed to start"):
            async with RobotProcess(broken, SESSION, store.shared_name):
                pass

    async def test_a_robot_that_disagrees_on_joints_is_refused(self, store: SessionStore) -> None:
        """Same check as in-process: the action vector follows the shape's order."""
        reversed_joints = _recipe(init_args={"joint_names": ["gripper", "shoulder_pan"], "position": 0.0})

        with pytest.raises(RuntimeError, match="failed to start"):
            async with RobotProcess(reversed_joints, SESSION, store.shared_name):
                pass

    async def test_the_worker_is_the_process(self, store: SessionStore) -> None:
        """Same shape as a ThreadedWorker: entering starts it, leaving ends it."""
        process = RobotProcess(_recipe(), SESSION, store.shared_name)

        async with process:
            assert process.is_alive()
            assert process.hz == 200.0

        assert not process.is_alive()

"""One store, two halves: scalars shared between processes, frames local."""

from __future__ import annotations

import numpy as np
import pytest

from runtimev2.features import FeatureSpec, camera_features, robot_features
from runtimev2.session_store import SessionStore
from runtimev2.store import UnknownFeatureError

JOINTS = ("shoulder_pan", "gripper")
OBS = "observation.arm.shoulder_pan.pos"
GRIP = "observation.arm.gripper.pos"
IMAGE = "observation.images.overhead"
FRAME = np.zeros((4, 4, 3), dtype=np.uint8)


def _spec() -> FeatureSpec:
    return FeatureSpec.build(robot_features("arm", JOINTS), camera_features({"overhead": (4, 4, 3)}))


@pytest.fixture
def store() -> SessionStore:
    created = SessionStore.create(_spec())
    try:
        yield created
    finally:
        created.close()


class TestRouting:
    def test_a_scalar_round_trips(self, store: SessionStore) -> None:
        store.write(OBS, 1.5, timestamp=2.0)

        sample = store.read(OBS)

        assert sample is not None
        assert sample.value == pytest.approx(1.5)

    def test_a_frame_round_trips(self, store: SessionStore) -> None:
        """Frames stay in this process; the publisher is where they come from."""
        store.write(IMAGE, FRAME, timestamp=3.0)

        sample = store.read(IMAGE)

        assert sample is not None
        assert sample.value.shape == (4, 4, 3)

    def test_a_scalar_reaches_shared_memory(self, store: SessionStore) -> None:
        """Which is what lets another process read it."""
        store.write(OBS, 9.0, timestamp=1.0)

        attached = SessionStore.attach(_spec(), store.shared_name)
        try:
            sample = attached.read(OBS)
        finally:
            attached.close()

        assert sample is not None
        assert sample.value == pytest.approx(9.0)

    def test_an_attached_store_sees_no_frames(self, store: SessionStore) -> None:
        """Deliberate: a recording process reads frames from the publisher."""
        store.write(IMAGE, FRAME, timestamp=1.0)

        attached = SessionStore.attach(_spec(), store.shared_name)
        try:
            assert attached.read(IMAGE) is None
        finally:
            attached.close()

    def test_an_unknown_feature_is_refused(self, store: SessionStore) -> None:
        with pytest.raises(UnknownFeatureError):
            store.write("observation.nope.wrist.pos", 1.0, timestamp=1.0)


class TestMixedAccess:
    def test_a_snapshot_spans_both_halves(self, store: SessionStore) -> None:
        """A dataset row and a policy observation both need scalars and frames."""
        store.write_many({OBS: 1.0, GRIP: 2.0}, timestamp=1.0)
        store.write(IMAGE, FRAME, timestamp=1.0)

        snapshot = store.snapshot((OBS, GRIP, IMAGE))

        assert set(snapshot) == {OBS, GRIP, IMAGE}

    def test_write_many_accepts_a_mixture(self, store: SessionStore) -> None:
        store.write_many({OBS: 1.0, IMAGE: FRAME}, timestamp=4.0)

        assert store.read(OBS) is not None
        assert store.read(IMAGE) is not None

    def test_a_full_snapshot_includes_everything_written(self, store: SessionStore) -> None:
        store.write(OBS, 1.0, timestamp=1.0)
        store.write(IMAGE, FRAME, timestamp=1.0)

        assert set(store.snapshot()) == {OBS, IMAGE}

    def test_unwritten_keys_are_absent(self, store: SessionStore) -> None:
        store.write(OBS, 1.0, timestamp=1.0)

        assert set(store.snapshot((OBS, GRIP, IMAGE))) == {OBS}
        assert not store.written((OBS, IMAGE))

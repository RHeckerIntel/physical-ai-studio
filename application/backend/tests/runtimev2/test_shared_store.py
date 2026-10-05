"""The same contract as FeatureStore, reachable from another process."""

from __future__ import annotations

import math
import multiprocessing as mp
import time

import pytest

from runtimev2.features import FeatureSpec, camera_features, robot_features
from runtimev2.shared_store import Layout, SharedFeatureStore, group_of
from runtimev2.store import UnknownFeatureError
from utils.multiprocessing import ensure_spawn_start_method

JOINTS = ("shoulder_pan", "gripper")
OBS = "observation.arm.shoulder_pan.pos"
GRIP = "observation.arm.gripper.pos"
ACT = "action.arm.shoulder_pan.pos"


def _spec(*, cameras: bool = False) -> FeatureSpec:
    parts = [robot_features("arm", JOINTS)]
    if cameras:
        parts.append(camera_features({"overhead": (480, 640, 3)}))
    return FeatureSpec.build(*parts)


@pytest.fixture
def store() -> SharedFeatureStore:
    created = SharedFeatureStore.create(_spec())
    try:
        yield created
    finally:
        created.close()


class TestLayout:
    def test_features_are_grouped_by_producer(self) -> None:
        """A group is what ``write_many`` publishes at one instant."""
        assert group_of(OBS) == "observation.arm"
        assert group_of(ACT) == "action.arm"

    def test_images_are_left_out(self) -> None:
        """They live in the camera publisher's memory already."""
        layout = Layout.build(_spec(cameras=True))

        assert not any("images" in key for key in layout.keys)

    def test_the_layout_is_derived_not_exchanged(self) -> None:
        """Two processes computing it from one spec must agree."""
        assert Layout.build(_spec()) == Layout.build(_spec())

    def test_the_block_is_sized_from_the_spec(self) -> None:
        layout = Layout.build(_spec())

        assert layout.nbytes == 8 * (len(layout.groups) + 2 * len(layout.keys))


class TestTheContract:
    def test_an_unwritten_feature_reads_as_nothing(self) -> None:
        """Not zero: a joint at its origin legitimately reports zero."""
        created = SharedFeatureStore.create(_spec())
        try:
            assert created.read(OBS) is None
            assert created.snapshot() == {}
        finally:
            created.close()

    def test_a_written_value_comes_back_with_its_timestamp(self, store: SharedFeatureStore) -> None:
        store.write(OBS, 1.25, timestamp=7.5)

        sample = store.read(OBS)

        assert sample is not None
        assert sample.value == pytest.approx(1.25)
        assert sample.timestamp == pytest.approx(7.5)

    def test_zero_is_a_real_value(self, store: SharedFeatureStore) -> None:
        store.write(OBS, 0.0, timestamp=1.0)

        assert store.read(OBS) is not None

    def test_an_unknown_feature_is_refused(self, store: SharedFeatureStore) -> None:
        with pytest.raises(UnknownFeatureError):
            store.write("observation.nope.wrist.pos", 1.0, timestamp=1.0)

    def test_an_image_feature_is_refused(self) -> None:
        """Rather than pretending to carry frames it cannot."""
        created = SharedFeatureStore.create(_spec(cameras=True))
        try:
            with pytest.raises(UnknownFeatureError):
                created.write("observation.images.overhead", 1.0, timestamp=1.0)
        finally:
            created.close()

    def test_snapshot_omits_unwritten_keys(self, store: SharedFeatureStore) -> None:
        store.write(OBS, 1.0, timestamp=1.0)

        assert set(store.snapshot((OBS, GRIP))) == {OBS}

    def test_written_reports_only_when_all_are_there(self, store: SharedFeatureStore) -> None:
        store.write(OBS, 1.0, timestamp=1.0)
        assert not store.written((OBS, GRIP))

        store.write(GRIP, 2.0, timestamp=1.0)
        assert store.written((OBS, GRIP))


def _writer(name: str, count: int) -> None:
    """Write a group repeatedly from another process."""
    store = SharedFeatureStore.attach(_spec(), name)
    try:
        for index in range(count):
            store.write_many({OBS: float(index), GRIP: float(index)}, timestamp=float(index))
            time.sleep(0.0005)
    finally:
        store.close()


class TestAcrossProcesses:
    def test_another_process_can_write_what_this_one_reads(self, store: SharedFeatureStore) -> None:
        ensure_spawn_start_method()
        process = mp.Process(target=_writer, args=(store.name, 50))
        process.start()
        process.join(timeout=30)

        assert process.exitcode == 0
        sample = store.read(OBS)
        assert sample is not None
        assert sample.value == pytest.approx(49.0)

    def test_a_reader_never_sees_a_half_written_group(self, store: SharedFeatureStore) -> None:
        """The property a mutex would also give, without a writer that can block."""
        ensure_spawn_start_method()
        process = mp.Process(target=_writer, args=(store.name, 400))
        process.start()
        try:
            torn = 0
            reads = 0
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and process.is_alive():
                snapshot = store.snapshot((OBS, GRIP))
                if len(snapshot) == 2:
                    reads += 1
                    # The writer sets both to the same number every time, so
                    # any difference is a read that caught it mid-write.
                    if snapshot[OBS].value != snapshot[GRIP].value:
                        torn += 1
        finally:
            process.join(timeout=30)

        assert reads > 20, f"only {reads} complete reads; the test proved nothing"
        assert torn == 0, f"{torn} of {reads} reads caught a half-written group"

    def test_attaching_with_a_mismatched_spec_is_refused(self, store: SharedFeatureStore) -> None:
        """Two processes disagreeing about the session is worth failing over."""
        wider = FeatureSpec.build(robot_features("arm", (*JOINTS, "elbow_flex", "wrist_roll", "wrist_flex")))

        with pytest.raises(ValueError, match="this session needs"):
            SharedFeatureStore.attach(wider, store.name)


def test_a_mid_write_sequence_yields_nothing_rather_than_garbage() -> None:
    """If a writer dies mid-write its group stays odd; a reader must not spin."""
    created = SharedFeatureStore.create(_spec())
    try:
        created.write_many({OBS: 1.0, GRIP: 1.0}, timestamp=1.0)
        # Leave the group looking mid-write, as a killed writer would.
        created._seq[created._group_index["observation.arm"]] += 1

        assert created.snapshot((OBS, GRIP)) == {}
        assert not math.isnan(created._stamps[created._index[OBS]])
    finally:
        created.close()

"""The same contract as FeatureStore, reachable from another process."""

from __future__ import annotations

import math
import multiprocessing as mp
import time

import pytest

from runtimev2.features import ACTION_KEY, STATE_KEY, FeatureSpec, camera_features, robot_features
from runtimev2.shared_store import Layout, SharedFeatureStore
from runtimev2.store import UnknownFeatureError
from utils.multiprocessing import ensure_spawn_start_method

JOINTS = ("shoulder_pan", "gripper")
OBS = STATE_KEY
ACT = ACTION_KEY


# Wide enough that a copy is not one indivisible memcpy, which is what makes
# tearing observable at all. The guarantee has to hold for any size; a robot's
# six joints are simply too few to catch a writer in the act.
WIDE = tuple(f"j{index}" for index in range(4096))


def _wide_spec() -> FeatureSpec:
    return FeatureSpec.build(robot_features(WIDE))


def _spec(*, cameras: bool = False) -> FeatureSpec:
    parts = [robot_features(JOINTS)]
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
    def test_a_robot_is_one_feature_per_kind(self) -> None:
        """Not one per joint: the vector is what every boundary speaks."""
        layout = Layout.build(_spec())

        assert set(layout.keys) == {OBS, ACT}
        assert layout.sizes == (len(JOINTS), len(JOINTS))

    def test_images_are_left_out(self) -> None:
        """They live in the camera publisher's memory already."""
        layout = Layout.build(_spec(cameras=True))

        assert not any("images" in key for key in layout.keys)

    def test_the_layout_is_derived_not_exchanged(self) -> None:
        """Two processes computing it from one spec must agree."""
        assert Layout.build(_spec()) == Layout.build(_spec())

    def test_the_block_is_sized_from_the_spec(self) -> None:
        layout = Layout.build(_spec())

        # A sequence and a timestamp per feature, plus every vector's numbers.
        assert layout.nbytes == 8 * (2 * len(layout.keys) + layout.total)

    def test_each_vector_gets_its_own_region(self) -> None:
        offsets = Layout.build(_spec()).offsets()

        starts = sorted(start for start, _ in offsets.values())
        assert starts == [0, len(JOINTS)]


class TestTheContract:
    def test_an_unwritten_feature_reads_as_nothing(self) -> None:
        """Not zero: a joint at its origin legitimately reports zero."""
        created = SharedFeatureStore.create(_spec())
        try:
            assert created.read(OBS) is None
            assert created.snapshot() == {}
        finally:
            created.close()

    def test_a_written_vector_comes_back_in_order(self, store: SharedFeatureStore) -> None:
        store.write(OBS, [1.25, 2.5], timestamp=7.5)

        sample = store.read(OBS)

        assert sample is not None
        assert list(sample.value) == pytest.approx([1.25, 2.5])
        assert sample.timestamp == pytest.approx(7.5)

    def test_zeros_are_real_values(self, store: SharedFeatureStore) -> None:
        store.write(OBS, [0.0, 0.0], timestamp=1.0)

        assert store.read(OBS) is not None

    def test_a_wrong_length_vector_is_refused(self, store: SharedFeatureStore) -> None:
        """The spec says how many joints a robot has; a mismatch is a bug."""
        with pytest.raises(ValueError, match="holds 2 values, got 3"):
            store.write(OBS, [1.0, 2.0, 3.0], timestamp=1.0)

    def test_an_unknown_feature_is_refused(self, store: SharedFeatureStore) -> None:
        with pytest.raises(UnknownFeatureError):
            store.write("observation.nope", [1.0, 2.0], timestamp=1.0)

    def test_an_image_feature_is_refused(self) -> None:
        """Rather than pretending to carry frames it cannot."""
        created = SharedFeatureStore.create(_spec(cameras=True))
        try:
            with pytest.raises(UnknownFeatureError):
                created.write("observation.images.overhead", [1.0], timestamp=1.0)
        finally:
            created.close()

    def test_snapshot_omits_unwritten_keys(self, store: SharedFeatureStore) -> None:
        store.write(OBS, [1.0, 1.0], timestamp=1.0)

        assert set(store.snapshot((OBS, ACT))) == {OBS}

    def test_written_reports_only_when_all_are_there(self, store: SharedFeatureStore) -> None:
        store.write(OBS, [1.0, 1.0], timestamp=1.0)
        assert not store.written((OBS, ACT))

        store.write(ACT, [2.0, 2.0], timestamp=1.0)
        assert store.written((OBS, ACT))


def _writer(name: str, count: int) -> None:
    """Write a robot's vector repeatedly from another process."""
    store = SharedFeatureStore.attach(_spec(), name)
    try:
        for index in range(count):
            store.write(OBS, [float(index)] * len(JOINTS), timestamp=float(index))
            time.sleep(0.0005)
    finally:
        store.close()


def _wide_writer(name: str, count: int) -> None:
    """Write a wide vector repeatedly, every component the same number."""
    store = SharedFeatureStore.attach(_wide_spec(), name)
    try:
        for index in range(count):
            store.write(STATE_KEY, [float(index)] * len(WIDE), timestamp=float(index))
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
        assert list(sample.value) == pytest.approx([49.0] * len(JOINTS))

    def test_a_reader_never_sees_a_half_written_vector(self) -> None:
        """The property a mutex would also give, without a writer that can block."""
        ensure_spawn_start_method()
        store = SharedFeatureStore.create(_wide_spec())
        process = mp.Process(target=_wide_writer, args=(store.name, 4000))
        process.start()
        try:
            torn = 0
            reads = 0
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and process.is_alive():
                sample = store.read(STATE_KEY)
                if sample is not None:
                    reads += 1
                    # Every component is written to the same number, so any
                    # difference is a read that caught the writer mid-write.
                    if len(set(sample.value.tolist())) != 1:
                        torn += 1
        finally:
            process.join(timeout=30)
            store.close()

        assert reads > 20, f"only {reads} complete reads; the test proved nothing"
        assert torn == 0, f"{torn} of {reads} reads caught a half-written vector"

    def test_attaching_with_a_mismatched_spec_is_refused(self, store: SharedFeatureStore) -> None:
        """Two processes disagreeing about the session is worth failing over."""
        wider = FeatureSpec.build(robot_features((*JOINTS, "elbow_flex", "wrist_roll", "wrist_flex")))

        with pytest.raises(ValueError, match="this session needs"):
            SharedFeatureStore.attach(wider, store.name)


def test_a_mid_write_sequence_yields_nothing_rather_than_garbage() -> None:
    """A writer that dies mid-write leaves its sequence odd; a reader must not
    spin on it, nor hand back a vector that could span two writes."""
    created = SharedFeatureStore.create(_spec())
    try:
        created.write(OBS, [1.0, 1.0], timestamp=1.0)
        # Leave it looking mid-write, as a killed writer would.
        created._seq[created._index[OBS]] += 1

        assert created.read(OBS) is None
        assert not math.isnan(created._stamps[created._index[OBS]])
    finally:
        created.close()

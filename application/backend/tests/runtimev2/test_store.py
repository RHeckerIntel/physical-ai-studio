"""The store is the session's current truth, and reading it cannot fail."""

from __future__ import annotations

import threading

import pytest

from runtimev2.features import ACTION_KEY, STATE_KEY, FeatureSpec, camera_features, robot_features
from runtimev2.store import FeatureStore, UnknownFeatureError

JOINTS = ["shoulder_pan", "gripper"]
OBS = STATE_KEY
ACT = ACTION_KEY
POSITIONS = [1.25, 2.5]


def _store() -> FeatureStore:
    return FeatureStore(FeatureSpec.build(robot_features(JOINTS), camera_features({"overhead": (480, 640, 3)})))


class TestReadWrite:
    def test_a_written_feature_reads_back_with_its_timestamp(self) -> None:
        store = _store()

        store.write(OBS, POSITIONS, timestamp=100.5)
        sample = store.read(OBS)

        assert sample is not None
        assert list(sample.value) == pytest.approx(POSITIONS)
        assert sample.timestamp == 100.5

    def test_an_unwritten_feature_is_none_rather_than_an_error(self) -> None:
        """Nothing having produced a value yet is a normal state at session start."""
        assert _store().read(OBS) is None

    def test_a_later_write_wins(self) -> None:
        store = _store()

        store.write(OBS, 1.0, timestamp=1.0)
        store.write(OBS, 2.0, timestamp=2.0)

        sample = store.read(OBS)
        assert sample is not None
        assert sample.value == 2.0

    def test_a_key_outside_the_spec_is_refused(self) -> None:
        """A mistyped key would otherwise sit unread, surfacing only as a missing model input."""
        store = _store()

        with pytest.raises(UnknownFeatureError):
            store.write("observation.follower.nonexistent.pos", 1.0, timestamp=1.0)

    def test_write_many_refuses_the_whole_batch(self) -> None:
        store = _store()

        with pytest.raises(UnknownFeatureError):
            store.write_many({OBS: 1.0, "bogus": 2.0}, timestamp=1.0)

        assert store.read(OBS) is None


class TestSnapshot:
    def test_a_snapshot_holds_only_what_was_written(self) -> None:
        store = _store()
        store.write(OBS, 1.0, timestamp=1.0)

        assert set(store.snapshot([OBS, ACT])) == {OBS}

    def test_a_snapshot_of_everything_needs_no_keys(self) -> None:
        store = _store()
        store.write_many({OBS: 1.0, ACT: 2.0}, timestamp=1.0)

        assert set(store.snapshot()) == {OBS, ACT}

    def test_written_reports_whether_every_key_has_a_value(self) -> None:
        store = _store()
        store.write(OBS, 1.0, timestamp=1.0)

        assert store.written([OBS])
        assert not store.written([OBS, ACT])

    def test_a_vector_is_never_seen_half_updated(self) -> None:
        """A reader must not catch a robot with some joints from this tick and some from the last.

        The reader sets the stop, not the writer: a fixed writer loop can finish
        before the reader is ever scheduled, which would pass while observing
        nothing.
        """
        store = _store()
        store.write_many({OBS: 0.0, ACT: 0.0}, timestamp=0.0)
        observations = 2000
        stop = threading.Event()

        def writer() -> None:
            seq = 0
            while not stop.is_set():
                seq += 1
                store.write_many({OBS: float(seq), ACT: float(seq)}, timestamp=float(seq))

        thread = threading.Thread(target=writer)
        thread.start()
        try:
            seen = [store.snapshot([OBS, ACT]) for _ in range(observations)]
        finally:
            stop.set()
            thread.join()

        pairs = [(shot[OBS].value, shot[ACT].value) for shot in seen if len(shot) == 2]
        assert len(pairs) == observations, "a snapshot lost a feature that had already been written"
        assert all(pan == grip for pan, grip in pairs), "a snapshot mixed two different ticks"
        assert {pan for pan, _ in pairs} != {0.0}, "the writer never interleaved with the reader"

"""Lifecycle guarantees of BaseProcessWorker that other workers rely on."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
from pathlib import Path

import pytest

from utils.multiprocessing import ensure_spawn_start_method
from workers.base import BaseProcessWorker, BaseThreadWorker

STARTED = "started"
TORN_DOWN = "torn-down"
_JOIN_TIMEOUT_S = 30.0


class _MarkerWorker(BaseProcessWorker):
    """Records the lifecycle stages it reaches, so a test can assert on teardown."""

    ROLE = "MarkerWorker"

    def __init__(self, markers: Path, *, stop_event: mp.synchronize.Event) -> None:
        super().__init__(stop_event=stop_event)
        self._markers = markers

    def _mark(self, name: str) -> None:
        (self._markers / name).touch()

    async def run_loop(self) -> None:
        self._mark(STARTED)
        # Blocking, like RuntimeSessionWorker's own loop: a signal has to land
        # on a thread that is not awaiting anything.
        while not self.should_stop():
            self.stop_aware_sleep(0.01)

    async def teardown(self) -> None:
        self._mark(TORN_DOWN)


def _wait_for(path: Path, timeout: float = _JOIN_TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise AssertionError(f"{path.name} never appeared")


@pytest.fixture
def running_worker(tmp_path: Path):
    """A spawned worker that has reached its run loop, joined on the way out."""
    # A spawn re-imports the backend, which is what resets the child's signal
    # handlers to their defaults -- the condition this module is about.
    ensure_spawn_start_method()
    worker = _MarkerWorker(tmp_path, stop_event=mp.Event())
    worker.start()
    try:
        _wait_for(tmp_path / STARTED)
        yield worker
    finally:
        if worker.is_alive():
            worker.kill()
        worker.join(timeout=_JOIN_TIMEOUT_S)


class TestShouldStop:
    def test_signal_flag_requests_a_stop(self) -> None:
        worker = _MarkerWorker(Path(), stop_event=mp.Event())
        assert not worker.should_stop()

        worker._handle_terminate(signal.SIGTERM, None)

        assert worker.should_stop()


class TestSigterm:
    def test_sigterm_runs_teardown(self, running_worker: _MarkerWorker, tmp_path: Path) -> None:
        """``Process.terminate()`` must not strand devices the worker holds.

        The default disposition ends the process between two bytecodes, so
        ``teardown()`` never runs and whatever the worker opened -- a camera
        publisher, a robot port -- is left to time out on its own, or not at all.
        """
        assert running_worker.pid is not None
        os.kill(running_worker.pid, signal.SIGTERM)
        running_worker.join(timeout=_JOIN_TIMEOUT_S)

        assert not running_worker.is_alive(), "worker did not exit after SIGTERM"
        assert (tmp_path / TORN_DOWN).exists(), "SIGTERM skipped teardown()"
        assert running_worker.exitcode == 0

    def test_stop_event_still_runs_teardown(self, running_worker: _MarkerWorker, tmp_path: Path) -> None:
        running_worker.request_stop()
        running_worker.join(timeout=_JOIN_TIMEOUT_S)

        assert not running_worker.is_alive()
        assert (tmp_path / TORN_DOWN).exists()
        assert running_worker.exitcode == 0

    def test_sigint_is_ignored(self, running_worker: _MarkerWorker, tmp_path: Path) -> None:
        """Ctrl+C hits the whole process group; the parent decides when children stop."""
        assert running_worker.pid is not None
        os.kill(running_worker.pid, signal.SIGINT)
        time.sleep(0.5)

        assert running_worker.is_alive(), "SIGINT should not stop a child worker"
        assert not (tmp_path / TORN_DOWN).exists()


class _ThreadMarkerWorker(BaseThreadWorker):
    """Thread-side twin of ``_MarkerWorker``."""

    ROLE = "ThreadMarkerWorker"

    def __init__(self, markers: Path, *, stop_event: mp.synchronize.Event) -> None:
        super().__init__(stop_event=stop_event)
        self._markers = markers

    async def run_loop(self) -> None:
        (self._markers / STARTED).touch()
        while not self.should_stop():
            self.stop_aware_sleep(0.01)

    async def teardown(self) -> None:
        (self._markers / TORN_DOWN).touch()


class TestThreadWorkerContextManager:
    def test_block_starts_and_stops_the_worker(self, tmp_path: Path) -> None:
        worker = _ThreadMarkerWorker(tmp_path, stop_event=mp.Event())

        with worker as entered:
            assert entered is worker
            _wait_for(tmp_path / STARTED)
            assert worker.is_alive()

        assert not worker.is_alive()
        assert (tmp_path / TORN_DOWN).exists()

    def test_the_worker_stops_when_the_body_raises(self, tmp_path: Path) -> None:
        """The device a worker holds has to be released on the error path too."""
        worker = _ThreadMarkerWorker(tmp_path, stop_event=mp.Event())

        with pytest.raises(ValueError, match="boom"), worker:
            _wait_for(tmp_path / STARTED)
            raise ValueError("boom")

        assert not worker.is_alive()


class TestStopPolling:
    def test_on_poll_runs_while_waiting(self, running_worker: _MarkerWorker) -> None:
        """A worker cannot exit while the parent stops draining its event queue."""
        calls = 0

        def poll() -> None:
            nonlocal calls
            calls += 1

        running_worker.stop(on_poll=poll)

        assert calls > 0, "stop() never drained while waiting"
        assert not running_worker.is_alive()

    def test_stop_is_idempotent(self, running_worker: _MarkerWorker) -> None:
        running_worker.stop()
        running_worker.stop()

        assert not running_worker.is_alive()
        assert running_worker.exitcode == 0

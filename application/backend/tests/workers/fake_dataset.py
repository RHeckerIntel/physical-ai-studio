"""A dataset worker that records into a JSON log instead of lerobot, for the recording tests.

Module-level so the spawned child can import it. The child writes the log; the
test reads it back.
"""

from __future__ import annotations

import json
from pathlib import Path

from workers.dataset_worker import DatasetWorker


class FakeMutation:
    def __init__(self, log: Path) -> None:
        self.log = log
        self.frames = 0

    def _write(self, event: str, **fields: object) -> None:
        with self.log.open("a") as log:
            log.write(json.dumps({"event": event, **fields}) + "\n")

    def add_frame(self, obs: dict, act: dict, task: str) -> None:
        if self.frames == 0:
            cameras = {key: list(value.shape) for key, value in obs.items() if hasattr(value, "shape")}
            joints = {key: value for key, value in obs.items() if not hasattr(value, "shape")}
            self._write("first_frame", task=task, joints=joints, action=act, cameras=cameras)
        self.frames += 1

    def save_episode(self) -> None:
        self._write("save", frames=self.frames)
        self.frames = 0

    def discard_buffer(self) -> None:
        self._write("discard", frames=self.frames)
        self.frames = 0

    def teardown(self) -> None:
        self._write("teardown")


class FakeDatasetWorker(DatasetWorker):
    def open_mutation(self) -> FakeMutation:  # type: ignore[override]
        return FakeMutation(Path(self.dataset.path) / "recording.jsonl")


def recording_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

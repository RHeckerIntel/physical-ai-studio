import time

import numpy as np

from runtimev2.control import ControlAlgorithm
from runtimev2.features import ACTION_KEY, STATE_KEY
from runtimev2.session_store import SessionStore


class MoveControl(ControlAlgorithm):
    """Command the follower to a specific position gradually over time."""

    def __init__(
        self,
        store: SessionStore,
        goal_position: list[float],
        goal_time: float,
        *,
        hz: float,
    ) -> None:
        super().__init__(store, name="move", hz=hz)
        self.goal_position = goal_position
        self.goal_time = goal_time

        sample = self._store.read(STATE_KEY)
        if sample is None:
            raise RuntimeError("the robot has not been read yet")
        self.start_position = np.asarray(sample.value, dtype=np.float32)
        self.start_time = time.monotonic()
        self.done = False
        print(f"goal_time: {goal_time}")
        print(f"goal_position: {goal_time}")

    def tick(self) -> None:
        sample = self._store.read(STATE_KEY)

        timestamp = time.monotonic()
        print(f"timestamp {timestamp}")
        print(f"sample {sample}")
        if sample is not None and timestamp - self.start_time < self.goal_time:
            fraction = (timestamp - self.start_time) / self.goal_time
            values = self.start_position + (self.goal_position - self.start_position) * np.float32(fraction)
            print(values)
            self._store.write(ACTION_KEY, values, timestamp=timestamp)
        else:
            self.done = True

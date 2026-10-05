"""One robot's loop, in its own process.

``RobotWorker``'s ticks, ramp and joint-order check, with
:class:`~runtimev2.workers.process.ProcessWorker` behind them instead of a
thread. Everything held here pickles -- shape, driver recipe, block name --
because spawning sends this object over; the driver and store are built in
``acquire``, since resolving a robot needs the database the parent has.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from runtimev2.workers.process import ProcessWorker

if TYPE_CHECKING:
    from collections.abc import Generator

    from runtimev2.environment import RobotShape, SessionShape

_REALTIME_PRIORITY = 10
"""Modest: enough to beat encoders, far below anything the kernel needs."""


@dataclass(frozen=True, slots=True)
class RobotRecipe:
    """Everything the child needs to be this robot, and nothing more.

    Attributes:
        shape: The robot's key and joint order, which its feature keys follow.
        driver: The driver's own configuration, as ``physicalai.config``
            serialises it -- the same recipe a runtime export carries. A plain
            dict, so it crosses a spawn boundary.
        hz: How often to read the hardware and command it.
    """

    shape: RobotShape
    driver: dict[str, Any]
    hz: float


def claim_priority() -> str:
    """Ask to be scheduled ahead of ordinary work, and report what happened.

    Best effort: real-time scheduling needs ``CAP_SYS_NICE``, and refusing to
    start over a scheduling favour would be worse than an occasional retry. The
    result is logged so a missed deadline can be read against what was granted;
    on this rig nothing is, and isolation alone proved enough.
    """
    try:
        os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(_REALTIME_PRIORITY))  # type: ignore[attr-defined]
    except (AttributeError, OSError, ValueError):
        pass
    else:
        return f"SCHED_RR at {_REALTIME_PRIORITY}"
    try:
        os.nice(-5)
    except OSError:
        return "ordinary scheduling (no privileges to raise priority)"
    else:
        return "niceness -5"


class RobotProcess(ProcessWorker):
    """Run one robot's loop in its own process, reading and writing the store."""

    def __init__(self, recipe: RobotRecipe, shape: SessionShape, shared_name: str) -> None:
        super().__init__(name=recipe.shape.key, hz=recipe.hz)
        self._recipe = recipe
        self._shape = shape
        self._shared_name = shared_name
        self._worker: Any = None  # built in the child; neither pickles

    @property
    def key(self) -> str:
        return self._recipe.shape.key

    @contextmanager
    def acquire(self) -> Generator[None]:
        """Attach the store, build the driver, and hold the robot.

        Runs in the child. Imports are local so the parent does not pay for the
        robot stack merely to be able to start one of these.
        """
        from physicalai.config import instantiate

        from runtimev2.session_store import SessionStore
        from runtimev2.workers.robot import RobotWorker

        logger.info("Robot {} process running with {}", self.key, claim_priority())
        store = SessionStore.attach(self._shape.feature_spec(), self._shared_name)
        try:
            driver = instantiate(self._recipe.driver)
            worker = RobotWorker(driver, store, shape=self._recipe.shape, hz=self._recipe.hz)  # type: ignore[arg-type]
            # Its own acquire connects and checks joint order, so the process
            # path duplicates none of it.
            with worker.acquire():
                self._worker = worker
                yield
        finally:
            self._worker = None
            store.close()

    def tick(self) -> None:
        """One pass of the robot loop -- the thread version's, unchanged.

        Raises:
            RuntimeError: Ticked before ``acquire``.
        """
        if self._worker is None:
            raise RuntimeError(f"Robot {self.key} is not acquired")
        self._worker.tick()

"""Tick something at a fixed rate, and notice when it cannot keep up.

Rates belong here rather than inside the workers. A worker's ``tick`` is one
pass with no timing in it, which is what lets a test step it without a clock
and lets the same worker run at 30Hz in one session and 100Hz in another.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable

_OVERRUN_REPORT_INTERVAL_S = 1.0


class Tickable(Protocol):
    """One pass of work, with a name to blame when it runs late."""

    @property
    def name(self) -> str: ...

    def tick(self) -> None: ...


def run_at(worker: Tickable, hz: float, should_stop: Callable[[], bool]) -> None:
    """Tick ``worker`` at ``hz`` until asked to stop.

    Whether the rate is being met is tracked here rather than in the store. A
    tick that overruns its period is the only thing that knows it did, and a
    reader finding an unchanged value cannot tell a slow producer from one
    whose value simply has not moved. Overruns are reported at most once per
    second -- at 100Hz, a line each would bury the signal in its own noise.
    """
    period = 1.0 / hz
    overruns = 0
    worst = 0.0
    next_report = time.monotonic() + _OVERRUN_REPORT_INTERVAL_S
    while not should_stop():
        started = time.monotonic()
        try:
            worker.tick()
        except Exception:
            # This runs on the worker's own thread, so an escaping exception
            # would otherwise kill it without a trace: the environment would
            # still look loaded while a device quietly stopped being read.
            # Re-raised after logging, because a worker that cannot tick has
            # nothing useful left to do.
            logger.exception("{} failed a tick and is stopping", worker.name)
            raise
        elapsed = time.monotonic() - started
        if elapsed > period:
            overruns += 1
            worst = max(worst, elapsed)
        else:
            time.sleep(period - elapsed)
        now = time.monotonic()
        if overruns and now >= next_report:
            logger.warning(
                "{} missed {} of its {:.0f}Hz deadlines, worst {:.1f}ms",
                worker.name,
                overruns,
                hz,
                worst * 1000,
            )
            overruns, worst = 0, 0.0
            next_report = now + _OVERRUN_REPORT_INTERVAL_S

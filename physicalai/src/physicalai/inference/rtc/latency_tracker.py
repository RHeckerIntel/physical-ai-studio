# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Sliding-window latency tracker for RTC inference delay estimation."""

from __future__ import annotations

import builtins
from collections import deque

import numpy as np


class LatencyTracker:
    """Track inference latency over a sliding window.

    Used to compute ``inference_delay = ceil(max_latency * fps)`` —
    a conservative estimate of how many actions will have been consumed
    by the time the next chunk arrives.

    Args:
        maxlen: Maximum number of samples in the sliding window.
    """

    def __init__(self, maxlen: int = 100) -> None:
        self._buffer: deque[float] = deque(maxlen=maxlen)

    def add(self, latency_s: float) -> None:
        """Record one inference latency measurement.

        Args:
            latency_s: Inference duration in seconds.
        """
        self._buffer.append(latency_s)

    def max(self) -> float:
        """Worst-case latency in the window. Returns 0 if empty."""
        return builtins.max(self._buffer) if self._buffer else 0.0

    def percentile(self, q: float) -> float:
        """Compute a percentile of recorded latencies.

        Args:
            q: Percentile in [0, 100].

        Returns:
            The q-th percentile, or 0.0 if no samples recorded.
        """
        if not self._buffer:
            return 0.0
        return float(np.percentile(np.array(self._buffer), q))

    def reset(self) -> None:
        """Clear all recorded latencies."""
        self._buffer.clear()

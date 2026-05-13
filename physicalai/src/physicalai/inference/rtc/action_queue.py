# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Thread-safe dual-track action queue for RTC inference.

Stores both original (normalized) actions for ``prev_chunk_left_over``
feedback and postprocessed (denormalized) actions for robot execution.
"""

from __future__ import annotations

import logging
from threading import Lock

import numpy as np

logger = logging.getLogger(__name__)


class RTCActionQueue:
    """Dual-track, thread-safe action queue for RTC.

    The queue stores two parallel tracks:
    - **original**: raw model output (normalized), used as
      ``prev_chunk_left_over`` for the next inference call.
    - **processed**: postprocessed (denormalized) actions sent to the
      robot.

    A cursor tracks consumption. ``get()`` pops one processed action.
    ``get_left_over()`` returns the unconsumed original tail.
    ``merge()`` replaces both tracks, trimming stale prefix actions.

    All public methods are thread-safe.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._original: np.ndarray | None = None
        self._processed: np.ndarray | None = None
        self._cursor: int = 0

    def get_action_index(self) -> int:
        """Current consumption index (snapshot for cross-check)."""
        with self._lock:
            return self._cursor

    def get(self) -> np.ndarray | None:
        """Pop one postprocessed action. Called from main thread.

        Returns:
            Action array of shape ``(action_dim,)``, or ``None`` if empty.
        """
        with self._lock:
            if self._processed is None or self._cursor >= len(self._processed):
                return None
            action = self._processed[self._cursor].copy()
            self._cursor += 1
            return action

    def get_left_over(self) -> np.ndarray | None:
        """Return unconsumed original (normalized) actions.

        Called from the RTC background thread to build
        ``prev_chunk_left_over`` for the next inference call.

        Returns:
            Array of shape ``(remaining, action_dim)`` or ``None``.
        """
        with self._lock:
            if self._original is None or self._cursor >= len(self._original):
                return None
            return self._original[self._cursor:].copy()


    def get_processed_left_over(self) -> np.ndarray | None:
        """Return unconsumed processed (denormalized) actions.

        Useful for blending or debugging.

        Returns:
            Array of shape ``(remaining, action_dim)`` or ``None``.
        """
        with self._lock:
            if self._processed is None or self._cursor >= len(self._processed):
                return None
            return self._processed[self._cursor:].copy()

    def merge(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        real_delay: int,
        action_index_before_inference: int | None = None,
    ) -> None:
        """Replace queue contents, trimming the first ``real_delay`` actions.

        Args:
            original: Raw model output, shape ``(chunk_size, action_dim)``.
            processed: Postprocessed actions, shape ``(chunk_size, action_dim)``.
            real_delay: Number of leading actions to discard (already
                executed during inference latency).
            action_index_before_inference: Cursor snapshot taken before
                inference started. Used to cross-check ``real_delay``
                against actual consumption.
        """
        with self._lock:
            # Cross-check: compare latency-based delay with actual consumed actions
            if action_index_before_inference is not None:
                actual_consumed = max(0, self._cursor - action_index_before_inference)
                if actual_consumed != real_delay:
                    logger.warning(
                        "RTC delay mismatch: real_delay=%d but actually consumed %d actions",
                        real_delay,
                        actual_consumed,
                    )

            # Clamp delay to prevent OOB slicing
            max_len = min(len(original), len(processed))
            clamped_delay = max(0, min(real_delay, max_len))
            if clamped_delay != real_delay:
                logger.warning(
                    "RTC delay clamped: real_delay=%d → %d (chunk_size=%d)",
                    real_delay,
                    clamped_delay,
                    max_len,
                )

            self._original = original[clamped_delay:]
            self._processed = processed[clamped_delay:]
            self._cursor = 0

    def qsize(self) -> int:
        """Number of unconsumed actions remaining."""
        with self._lock:
            if self._processed is None:
                return 0
            return max(0, len(self._processed) - self._cursor)

    def reset(self) -> None:
        """Clear all state."""
        with self._lock:
            self._original = None
            self._processed = None
            self._cursor = 0

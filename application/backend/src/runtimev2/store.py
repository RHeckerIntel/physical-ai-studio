"""The session's current truth: the latest value of every feature.

One store per session holds what is true right now. Device-owning workers write
into it at whatever rate their hardware runs; readers -- a dataset being
recorded, a model, the websocket stream -- take what they need at whatever rate
suits them. Nothing here talks to a device or a transport, so reading the truth
cannot block on hardware and cannot fail.

Rates are deliberately not coordinated. A robot at 100Hz, cameras at 30 and a
recording at 50 all work, because every value carries the producer's own
timestamp and a reader decides what to do about the skew rather than having it
hidden. ``time.monotonic`` is system-wide on Linux, so timestamps from
different processes are directly comparable.

Values are assumed to arrive in order, and a reader is assumed to want the
latest one. Whether a producer is keeping up is the producer's business -- its
loop knows its own deadline -- so there is nothing here for a reader to use to
detect a gap.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from runtimev2.features import FeatureSpec


@dataclass(frozen=True, slots=True)
class Sample:
    """One feature's value, as of when its producer measured it.

    Attributes:
        value: The feature's vector -- a robot's joint positions in driver
            order, or a camera frame.
        timestamp: The producer's ``time.monotonic()`` at the moment of
            measurement -- not when it reached the store, which would fold
            transport delay into the reading.
    """

    value: Any
    timestamp: float


class UnknownFeatureError(KeyError):
    """Raised when a key is not in the session's spec.

    A typo in a feature key would otherwise sit in the store unread, and the
    only symptom would be a model or recording quietly missing an input.
    """


class FeatureStore:
    """Latest ``Sample`` per feature, for one session.

    Thread-safe: writers arrive from worker threads while readers snapshot.
    """

    def __init__(self, spec: FeatureSpec) -> None:
        self._spec = spec
        self._samples: dict[str, Sample] = {}
        self._lock = threading.Lock()

    @property
    def spec(self) -> FeatureSpec:
        return self._spec

    def write(self, key: str, value: Any, *, timestamp: float) -> None:
        """Record ``key``'s current value.

        Raises:
            UnknownFeatureError: ``key`` is not in this session's spec.
        """
        if key not in self._spec:
            raise UnknownFeatureError(key)
        with self._lock:
            self._samples[key] = Sample(value, timestamp)

    def write_many(self, values: Mapping[str, Any], *, timestamp: float) -> None:
        """Record several features measured at the same moment.

        One lock acquisition, so a reader cannot catch half of a robot's joints
        updated and half not.

        Raises:
            UnknownFeatureError: Any key is not in this session's spec.
        """
        unknown = [key for key in values if key not in self._spec]
        if unknown:
            raise UnknownFeatureError(", ".join(sorted(unknown)))
        samples = {key: Sample(value, timestamp) for key, value in values.items()}
        with self._lock:
            self._samples.update(samples)

    def read(self, key: str) -> Sample | None:
        """Return ``key``'s latest sample, or ``None`` if nothing has written it yet."""
        with self._lock:
            return self._samples.get(key)

    def snapshot(self, keys: Collection[str] | None = None) -> dict[str, Sample]:
        """Return the latest sample of every requested feature that has one.

        Taken under one lock, so the result is a consistent view rather than
        something assembled across several reads. Keys nobody has written are
        absent rather than ``None``, so a caller that needs them all can check
        with ``len``.
        """
        with self._lock:
            if keys is None:
                return dict(self._samples)
            return {key: self._samples[key] for key in keys if key in self._samples}

    def written(self, keys: Collection[str]) -> bool:
        """Whether every one of ``keys`` has been written at least once."""
        with self._lock:
            return all(key in self._samples for key in keys)

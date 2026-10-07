"""The session's scalar truth, in memory every process can reach.

Same contract as :class:`~runtimev2.store.FeatureStore`, backed by shared
memory so a robot loop can run in its own process -- which is what keeps it
from being descheduled into a serial timeout.

Scalars only: frames already live in the camera publisher's shared memory, so
image features are refused rather than copied through a second channel.

Readers and writers coordinate with a seqlock per feature, not a mutex. A
writer must not be able to stall the robot loop, and one that *dies* mid-write
must not leave a lock held. This assumes stores to adjacent cells become
visible in program order, which holds on x86-64 and arm64; Python offers no
portable barrier, and a dead mutex holder wedging every reader is the worse
failure.

A feature is a whole vector, so atomicity is per feature and there is nothing
to group: a reader either sees a robot's previous positions or its next ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any

import numpy as np

from runtimev2.store import Sample, UnknownFeatureError

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from runtimev2.features import FeatureSpec

_WORD = 8
"""Bytes per cell. Sequences are int64, values and timestamps float64."""

_READ_ATTEMPTS = 8
"""Retries before a reader gives up on a group that keeps changing under it.

Only reachable if a writer is descheduled mid-write repeatedly, or died in the
middle of one. Returning what we have beats spinning in a robot's loop.
"""


@dataclass(frozen=True, slots=True)
class Layout:
    """Where each feature's vector and sequence live in the block.

    Fixed by the spec, so another process maps it knowing only the session.
    """

    keys: tuple[str, ...]
    sizes: tuple[int, ...]
    """How many numbers each key holds, in the same order."""

    @classmethod
    def build(cls, spec: FeatureSpec) -> Layout:
        """Lay out ``spec``'s vectors, sorted so two processes agree without
        exchanging anything."""
        # FeatureSpec.keys() is its own method, not a mapping's.
        keys = tuple(sorted(key for key in spec.keys() if not spec[key].is_image))  # noqa: SIM118
        return cls(keys=keys, sizes=tuple(spec[key].size for key in keys))

    @property
    def total(self) -> int:
        """Numbers across every vector."""
        return sum(self.sizes)

    @property
    def nbytes(self) -> int:
        # A sequence and a timestamp per feature, then every vector's numbers.
        return _WORD * (2 * len(self.keys) + self.total)

    def offsets(self) -> dict[str, tuple[int, int]]:
        """Each key's ``(start, size)`` within the values region."""
        offsets: dict[str, tuple[int, int]] = {}
        start = 0
        for key, size in zip(self.keys, self.sizes, strict=True):
            offsets[key] = (start, size)
            start += size
        return offsets

    def index_of(self) -> dict[str, int]:
        return {key: index for index, key in enumerate(self.keys)}


class SharedFeatureStore:
    """Latest vector per feature, in shared memory.

    :meth:`create` here, :meth:`attach` from another process with the same spec.
    """

    def __init__(self, spec: FeatureSpec, block: shared_memory.SharedMemory, *, owner: bool) -> None:
        self._spec = spec
        self._layout = Layout.build(spec)
        self._block = block
        self._owner = owner
        self._index = self._layout.index_of()
        self._offsets = self._layout.offsets()

        count = len(self._layout.keys)
        buffer = self._block.buf
        self._seq = np.ndarray((count,), dtype=np.int64, buffer=buffer, offset=0)
        self._stamps = np.ndarray((count,), dtype=np.float64, buffer=buffer, offset=_WORD * count)
        self._values = np.ndarray((self._layout.total,), dtype=np.float64, buffer=buffer, offset=_WORD * 2 * count)

    @classmethod
    def create(cls, spec: FeatureSpec) -> SharedFeatureStore:
        """Allocate a block for ``spec`` and mark every feature unwritten."""
        layout = Layout.build(spec)
        block = shared_memory.SharedMemory(create=True, size=max(layout.nbytes, _WORD))
        store = cls(spec, block, owner=True)
        store._seq[:] = 0
        store._values[:] = 0.0
        # NaN means unwritten: a joint at its origin reports a real zero.
        store._stamps[:] = math.nan
        return store

    @classmethod
    def attach(cls, spec: FeatureSpec, name: str) -> SharedFeatureStore:
        """Map a block another process created, for the same ``spec``.

        Raises:
            ValueError: Wrong size, so the two disagree about the session.
        """
        block = shared_memory.SharedMemory(name=name)
        expected = max(Layout.build(spec).nbytes, _WORD)
        if block.size < expected:
            block.close()
            raise ValueError(f"shared store {name} holds {block.size} bytes, this session needs {expected}")
        return cls(spec, block, owner=False)

    @property
    def name(self) -> str:
        """The block's name, for another process to attach by."""
        return self._block.name

    @property
    def spec(self) -> FeatureSpec:
        return self._spec

    def write(self, key: str, values: Any, *, timestamp: float) -> None:  # an array or a sequence
        """Record ``key``'s current vector.

        Atomic: a reader sees either the previous vector or this one, never a
        mixture, which is what a robot's joints moving together requires.

        Raises:
            UnknownFeatureError: Not a vector feature of this session.
            ValueError: Wrong number of components for this feature.
        """
        index = self._index.get(key)
        if index is None:
            raise UnknownFeatureError(key)
        start, size = self._offsets[key]
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.size != size:
            raise ValueError(f"{key} holds {size} values, got {vector.size}")

        self._seq[index] += 1  # odd: a write is in progress
        try:
            self._values[start : start + size] = vector
            self._stamps[index] = timestamp
        finally:
            self._seq[index] += 1  # even again: readers may proceed

    def write_many(self, values: Mapping[str, Any], *, timestamp: float) -> None:
        """Record several features measured at the same moment.

        Atomic per feature, which is the granularity a producer writes at --
        no device reports two of them at once.

        Raises:
            UnknownFeatureError: Any key is not a vector of this session.
        """
        unknown = [key for key in values if key not in self._index]
        if unknown:
            raise UnknownFeatureError(", ".join(sorted(unknown)))
        for key, vector in values.items():
            self.write(key, vector, timestamp=timestamp)

    def read(self, key: str) -> Sample | None:
        """Return ``key``'s latest vector, or ``None`` if nothing has written it."""
        index = self._index.get(key)
        if index is None:
            return None
        start, size = self._offsets[key]
        for _ in range(_READ_ATTEMPTS):
            before = int(self._seq[index])
            if before % 2:
                continue  # a write is in progress
            values = np.array(self._values[start : start + size], dtype=np.float32)
            timestamp = float(self._stamps[index])
            if int(self._seq[index]) != before:
                continue  # it moved under us; the vector could span two writes
            return None if math.isnan(timestamp) else Sample(values, timestamp)
        return None

    def snapshot(self, keys: Collection[str] | None = None) -> dict[str, Sample]:
        """Latest vector of every requested feature that has one.

        Unwritten keys are absent rather than ``None``.
        """
        wanted = self._layout.keys if keys is None else keys
        samples = {}
        for key in wanted:
            sample = self.read(key)
            if sample is not None:
                samples[key] = sample
        return samples

    def written(self, keys: Collection[str]) -> bool:
        """Whether every one of ``keys`` has been written at least once."""
        snapshot = self.snapshot(keys)
        return all(key in snapshot for key in keys)

    def close(self) -> None:
        """Release this process's mapping, and the block if we created it."""
        self._seq = self._values = self._stamps = None  # type: ignore[assignment]
        self._block.close()
        if self._owner:
            self._block.unlink()

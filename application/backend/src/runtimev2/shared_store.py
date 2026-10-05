"""The session's scalar truth, in memory every process can reach.

Same contract as :class:`~runtimev2.store.FeatureStore`, backed by shared
memory so a robot loop can run in its own process -- which is what keeps it
from being descheduled into a serial timeout.

Scalars only: frames already live in the camera publisher's shared memory, so
image features are refused rather than copied through a second channel.

Readers and writers coordinate with a seqlock per group, not a mutex. A writer
must not be able to stall the robot loop, and one that *dies* mid-write must
not leave a lock held. This assumes stores to adjacent cells become visible in
program order, which holds on x86-64 and arm64; Python offers no portable
barrier, and a dead mutex holder wedging every reader is the worse failure.
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


def group_of(key: str) -> str:
    """The group a feature is written as part of: its first two segments.

    ``write_many`` is always one producer publishing one device at one instant,
    which is the granularity a reader must not see split.
    """
    parts = key.split(".", 2)
    return ".".join(parts[:2])


@dataclass(frozen=True, slots=True)
class Layout:
    """Where each feature and group lives in the block.

    Fixed by the spec, so another process maps it knowing only the session.
    """

    keys: tuple[str, ...]
    groups: tuple[str, ...]

    @classmethod
    def build(cls, spec: FeatureSpec) -> Layout:
        """Lay out ``spec``'s scalars, sorted so two processes agree without
        exchanging anything."""
        # FeatureSpec.keys() is its own method, not a mapping's.
        keys = tuple(sorted(key for key in spec.keys() if not spec[key].is_image))  # noqa: SIM118
        groups = tuple(sorted({group_of(key) for key in keys}))
        return cls(keys=keys, groups=groups)

    @property
    def nbytes(self) -> int:
        return _WORD * (len(self.groups) + 2 * len(self.keys))

    def index_of(self) -> dict[str, int]:
        return {key: index for index, key in enumerate(self.keys)}

    def group_index_of(self) -> dict[str, int]:
        return {group: index for index, group in enumerate(self.groups)}


class SharedFeatureStore:
    """Latest sample per scalar feature, in shared memory.

    :meth:`create` here, :meth:`attach` from another process with the same spec.
    """

    def __init__(self, spec: FeatureSpec, block: shared_memory.SharedMemory, *, owner: bool) -> None:
        self._spec = spec
        self._layout = Layout.build(spec)
        self._block = block
        self._owner = owner
        self._index = self._layout.index_of()
        self._group_index = self._layout.group_index_of()
        self._group_of_index = [self._group_index[group_of(key)] for key in self._layout.keys]

        groups, keys = len(self._layout.groups), len(self._layout.keys)
        buffer = self._block.buf
        self._seq = np.ndarray((groups,), dtype=np.int64, buffer=buffer, offset=0)
        self._values = np.ndarray((keys,), dtype=np.float64, buffer=buffer, offset=_WORD * groups)
        self._stamps = np.ndarray((keys,), dtype=np.float64, buffer=buffer, offset=_WORD * (groups + keys))

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

    def write(self, key: str, value: Any, *, timestamp: float) -> None:  # mirrors FeatureStore
        """Record ``key``'s current value.

        Raises:
            UnknownFeatureError: Not a scalar feature of this session.
        """
        self.write_many({key: value}, timestamp=timestamp)

    def write_many(self, values: Mapping[str, Any], *, timestamp: float) -> None:
        """Record several features measured at the same moment.

        Atomic per group, so a reader cannot catch half a robot's joints
        updated. Across groups it is only as consistent as the producers were.

        Raises:
            UnknownFeatureError: Any key is not a scalar of this session.
        """
        unknown = [key for key in values if key not in self._index]
        if unknown:
            raise UnknownFeatureError(", ".join(sorted(unknown)))
        indices = [self._index[key] for key in values]
        groups = sorted({self._group_of_index[index] for index in indices})

        for group in groups:
            self._seq[group] += 1  # odd: a write is in progress
        try:
            for index, value in zip(indices, values.values(), strict=True):
                self._values[index] = float(value)
                self._stamps[index] = timestamp
        finally:
            for group in groups:
                self._seq[group] += 1  # even again: readers may proceed

    def read(self, key: str) -> Sample | None:
        """Return ``key``'s latest sample, or ``None`` if nothing has written it."""
        return self.snapshot((key,)).get(key)

    def snapshot(self, keys: Collection[str] | None = None) -> dict[str, Sample]:
        """Latest sample of every requested feature that has one.

        Consistent per group. Unwritten keys are absent rather than ``None``.
        """
        wanted = self._layout.keys if keys is None else tuple(key for key in keys if key in self._index)
        by_group: dict[int, list[str]] = {}
        for key in wanted:
            by_group.setdefault(self._group_of_index[self._index[key]], []).append(key)

        samples: dict[str, Sample] = {}
        for group, group_keys in by_group.items():
            samples.update(self._read_group(group, group_keys))
        return samples

    def _read_group(self, group: int, keys: list[str]) -> dict[str, Sample]:
        """Read one group between two equal, even sequence numbers.

        A moved sequence means a writer was mid-write, so the values could span
        two instants and are read again.
        """
        indices = [self._index[key] for key in keys]
        for _ in range(_READ_ATTEMPTS):
            before = int(self._seq[group])
            if before % 2:
                continue
            values = [float(self._values[index]) for index in indices]
            stamps = [float(self._stamps[index]) for index in indices]
            if int(self._seq[group]) != before:
                continue
            return {
                key: Sample(value, timestamp)
                for key, value, timestamp in zip(keys, values, stamps, strict=True)
                if not math.isnan(timestamp)
            }
        return {}

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

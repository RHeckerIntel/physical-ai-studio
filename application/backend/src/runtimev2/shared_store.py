"""The session's scalar truth, in memory every process can reach.

Same contract as :class:`~runtimev2.store.FeatureStore` -- latest value per
feature, each carrying its producer's timestamp -- but backed by a block of
shared memory, so a robot can run in its own process and still be read by a
recording in another.

That matters for timing rather than throughput. A robot loop has to collect a
serial reply before the driver's packet timeout; being descheduled costs a
retry worth tens of milliseconds, which arrives in a recorded episode as a
frame that is *misdated* rather than dropped, because LeRobot timestamps frames
by index. Keeping the loop in its own process is what lets it be scheduled
apart from image encoding and inference.

Scalars only. Images already live in the camera publisher's shared memory, so
they do not need copying through here; a process that wants frames attaches to
the publisher. :class:`SharedFeatureStore` therefore refuses image features
rather than pretending to carry them.

Writers never block
-------------------
Readers and writers coordinate with a seqlock per group rather than a mutex: a
writer that is descheduled mid-write must not be able to stall the robot loop,
and a writer that *dies* mid-write must not leave a lock held forever. A writer
bumps its group's sequence to odd, writes, and bumps it to even; a reader takes
the sequence, reads, and retries if it moved. So a write is atomic as far as
any reader can tell -- nobody sees half of a robot's joints updated -- without
anyone waiting on anyone.

This relies on stores to adjacent ``int64``/``float64`` cells becoming visible
in program order, which holds on the x86-64 and arm64 targets this runs on.
There is no portable memory barrier available from Python; the alternative is a
cross-process mutex, whose failure mode -- a dead holder wedging every reader
-- is worse than the one being avoided.
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
    """The group a feature is written as part of.

    The first two segments: ``observation.follower.wrist.pos`` belongs to
    ``observation.follower``. Groups exist because ``write_many`` is always one
    producer publishing one device's features at one instant, and that is
    exactly the granularity a reader must not see split.
    """
    parts = key.split(".", 2)
    return ".".join(parts[:2])


@dataclass(frozen=True, slots=True)
class Layout:
    """Where each feature and group lives in the block.

    Fixed when a session is described, which is what makes the block a fixed
    size and lets another process map it knowing only the spec.
    """

    keys: tuple[str, ...]
    groups: tuple[str, ...]

    @classmethod
    def build(cls, spec: FeatureSpec) -> Layout:
        """Lay out every scalar feature in ``spec``, grouped by producer.

        Sorted, so two processes computing the layout from the same spec agree
        without having to exchange it.
        """
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

    Create one with :meth:`create` and reach it from another process with
    :meth:`attach`, passing the name and the same spec.
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
        # NaN is "nobody has written this": distinguishable from a real zero,
        # which a joint at its origin legitimately reports.
        store._stamps[:] = math.nan
        return store

    @classmethod
    def attach(cls, spec: FeatureSpec, name: str) -> SharedFeatureStore:
        """Map a block another process created, for the same ``spec``.

        Raises:
            ValueError: The block is not the size this spec implies, which
                means the two processes disagree about the session.
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

        Atomic per group, so a reader cannot catch half of a robot's joints
        updated. Writing across groups bumps each one, which is as consistent
        as the producers were -- nothing here invents a wider guarantee.

        Raises:
            UnknownFeatureError: Any key is not a scalar feature of this session.
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
        """Return the latest sample of every requested feature that has one.

        Consistent per group: each group is read between two equal, even
        sequence numbers, so the values in it were written together. Keys
        nobody has written are absent rather than ``None``.
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
        """Read one group's keys between two equal, even sequence numbers.

        An odd sequence, or one that moved while we read, means a writer was
        mid-write: the values may be from two different instants, so they are
        discarded and read again.
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

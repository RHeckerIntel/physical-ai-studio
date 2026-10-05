"""The session's truth, split by what can usefully cross a process boundary.

Scalars go in shared memory, so a robot running in its own process can publish
what it measured and read what it is being asked to do. Frames do not: they
already sit in the camera publisher's shared memory, and copying a megabyte per
camera per tick through a second channel would buy nothing. A process that
wants frames attaches to the publisher.

Callers see one store with one interface and need not care which half a feature
lives in; the spec already says, because an image feature is marked as one.

The split has a consequence worth knowing: a child process attaching to this
store sees scalars but no frames. That is deliberate rather than a gap -- a
recording process reads its own frames from the publisher, which is also how it
gets them without a copy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from runtimev2.shared_store import SharedFeatureStore
from runtimev2.store import FeatureStore, UnknownFeatureError

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from runtimev2.features import FeatureSpec
    from runtimev2.store import Sample


class SessionStore:
    """Latest sample per feature: scalars shared between processes, frames local."""

    def __init__(self, spec: FeatureSpec, shared: SharedFeatureStore) -> None:
        self._spec = spec
        self._shared = shared
        self._images = FeatureStore(spec)

    @classmethod
    def create(cls, spec: FeatureSpec) -> SessionStore:
        """Allocate this session's shared block and an empty frame store."""
        return cls(spec, SharedFeatureStore.create(spec))

    @classmethod
    def attach(cls, spec: FeatureSpec, name: str) -> SessionStore:
        """Reach a session's scalars from another process.

        The returned store carries no frames: nothing in this process has
        written any, and the publisher is where they come from.

        Raises:
            ValueError: The block is not the size this spec implies.
        """
        return cls(spec, SharedFeatureStore.attach(spec, name))

    @property
    def spec(self) -> FeatureSpec:
        return self._spec

    @property
    def shared_name(self) -> str:
        """The scalar block's name, for another process to attach by."""
        return self._shared.name

    def write(self, key: str, value: Any, *, timestamp: float) -> None:  # a float or a frame
        """Record ``key``'s current value.

        Raises:
            UnknownFeatureError: ``key`` is not in this session's spec.
        """
        if key not in self._spec:
            raise UnknownFeatureError(key)
        target = self._images if self._spec[key].is_image else self._shared
        target.write(key, value, timestamp=timestamp)

    def write_many(self, values: Mapping[str, Any], *, timestamp: float) -> None:
        """Record several features measured at the same moment.

        Atomic within each half. A producer writes one device's features, and
        no device produces both frames and scalars, so nothing needs a
        guarantee spanning the two.

        Raises:
            UnknownFeatureError: Any key is not in this session's spec.
        """
        unknown = [key for key in values if key not in self._spec]
        if unknown:
            raise UnknownFeatureError(", ".join(sorted(unknown)))
        images = {key: value for key, value in values.items() if self._spec[key].is_image}
        scalars = {key: value for key, value in values.items() if key not in images}
        if images:
            self._images.write_many(images, timestamp=timestamp)
        if scalars:
            self._shared.write_many(scalars, timestamp=timestamp)

    def read(self, key: str) -> Sample | None:
        """Return ``key``'s latest sample, or ``None`` if nothing has written it."""
        if key not in self._spec:
            return None
        target = self._images if self._spec[key].is_image else self._shared
        return target.read(key)

    def snapshot(self, keys: Collection[str] | None = None) -> dict[str, Sample]:
        """Return the latest sample of every requested feature that has one.

        Consistent within each half, which is the granularity producers write
        at. Keys nobody has written are absent rather than ``None``.
        """
        if keys is None:
            return {**self._shared.snapshot(), **self._images.snapshot()}
        wanted = [key for key in keys if key in self._spec]
        images = [key for key in wanted if self._spec[key].is_image]
        scalars = [key for key in wanted if key not in set(images)]
        return {**self._shared.snapshot(scalars), **self._images.snapshot(images)}

    def written(self, keys: Collection[str]) -> bool:
        """Whether every one of ``keys`` has been written at least once."""
        snapshot = self.snapshot(keys)
        return all(key in snapshot for key in keys)

    def close(self) -> None:
        """Release the shared block. Frames are dropped with this object."""
        self._shared.close()

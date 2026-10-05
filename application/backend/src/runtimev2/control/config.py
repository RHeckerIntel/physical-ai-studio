"""What to drive with, as a value rather than a name.

A control's rate is part of choosing it, not a separate argument, so a client
asks for one thing and the state reports the same thing back. Options a control
grows later -- running inference asynchronously, how to blend two inputs --
belong on its config, where they can be reported and recorded.

These carry no behaviour and hold no devices. The session matches on them to
build the control, because only the session has the store and the leader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

# A leader is read as fast as it will answer, which is what makes teleoperation
# feel direct.
DEFAULT_TELEOP_HZ = 100.0
# What a policy emits at. Inference runs every ``chunk_size`` emits, so this is
# not the rate the accelerator sees.
DEFAULT_MODEL_HZ = 30.0


@dataclass(frozen=True, slots=True)
class TeleopControlConfig:
    """Drive the follower from the leader arm."""

    hz: float = DEFAULT_TELEOP_HZ
    """How often the leader is read. Each reading is one command."""

    kind: ClassVar[str] = "teleop"


@dataclass(frozen=True, slots=True)
class ModelControlConfig:
    """Drive the follower from a loaded policy."""

    hz: float = DEFAULT_MODEL_HZ
    """How often an action is emitted, not how often inference runs."""

    kind: ClassVar[str] = "model"


ControlConfig = TeleopControlConfig | ModelControlConfig

_BY_KIND: dict[str, type[TeleopControlConfig | ModelControlConfig]] = {
    TeleopControlConfig.kind: TeleopControlConfig,
    ModelControlConfig.kind: ModelControlConfig,
}


def describe(config: ControlConfig) -> dict[str, Any]:
    """Render a config for a client, kind included."""
    return {"kind": config.kind, "hz": config.hz}


def parse(payload: Any) -> ControlConfig | None:  # whatever a client sent
    """Read a config from a client message, or ``None`` to drive with nothing.

    A bare kind is accepted as well as a mapping, because asking for the
    default rate should not mean naming it.

    Raises:
        ValueError: Not a control this build knows, or a rate that is not a
            positive number.
    """
    if payload is None:
        return None
    fields: dict[str, Any] = {}
    if isinstance(payload, str):
        kind = payload
    else:
        fields = dict(payload)
        kind = str(fields.pop("kind", ""))
    config_type = _BY_KIND.get(kind)
    if config_type is None:
        raise ValueError(f"Unknown control {kind!r}; expected one of {sorted(_BY_KIND)}")
    options: dict[str, Any] = {}
    if "hz" in fields:
        hz = float(fields.pop("hz"))
        if hz <= 0:
            raise ValueError(f"A control rate must be positive, got {hz}")
        options["hz"] = hz
    # Whatever is left was asked for and not understood, which is worth saying
    # rather than silently dropping.
    if fields:
        raise ValueError(f"Unknown options for control {kind!r}: {sorted(fields)}")
    return config_type(**options)

"""The controls a session can run, and building one from its config. See ``base.py``."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from runtime_session.controls.base import Control
from runtime_session.controls.hil import HilControl
from runtime_session.controls.model import ModelControl
from runtime_session.controls.teleop import TeleopControl
from schemas.runtime_session import HilControlConfig, ModelControlConfig

if TYPE_CHECKING:
    from physicalai.robot import Robot

    from runtime_session.bus import Bus
    from runtime_session.model import ProcessModel
    from runtime_session.store import Store
    from schemas.runtime_session import ControlConfig

__all__ = ["Control", "HilControl", "ModelControl", "TeleopControl", "build_control"]


def build_control(
    config: ControlConfig,
    store: Store,
    leader: Robot | None,
    model: ProcessModel | None = None,
    bus: Bus | None = None,
) -> Control:
    """Raises ``ValueError`` when the session cannot run ``config``.

    A model or human-in-the-loop control needs the loaded ``model``. A control serves its own commands on ``bus``.
    """
    if isinstance(config, ModelControlConfig | HilControlConfig) and model is None:
        raise ValueError("Load a model first")
    if isinstance(config, ModelControlConfig):
        return ModelControl(store, cast("ProcessModel", model), config)
    if leader is None:
        raise ValueError("Teleoperation needs a connected leader arm; connect one and reload the environment")
    if isinstance(config, HilControlConfig):
        return HilControl(store, cast("ProcessModel", model), leader, config, bus)
    return TeleopControl(store, leader, config, bus)

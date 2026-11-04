from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast

from physicalai.robot import TorqueControllable

from exceptions import RuntimeSessionCommandError
from runtime_session.bus import NoResponderError, command_key
from runtime_session.controls.base import Control
from runtime_session.controls.model import ModelControl
from runtime_session.controls.teleop import TeleopControl
from runtime_session.store import ACTION, STATE
from schemas.runtime_session import (
    CommandDone,
    InterruptCommand,
    ResumeCommand,
    SaveEpisodeCommand,
    StartEpisodeCommand,
    TakeOverCommand,
    TeleopControlConfig,
)
from workers.dataset_worker import RecordingStatus

if TYPE_CHECKING:
    import numpy as np
    from physicalai.robot import Robot

    from runtime_session.bus import Bus
    from runtime_session.controls.base import Commands
    from runtime_session.model import ProcessModel
    from runtime_session.store import Store
    from schemas.runtime_session import HilControlConfig, HilPhase

_EPISODE_S = 300.0
"""How long starting or saving an episode may take; saving encodes its videos."""


class HilControl(Control):
    """Human in the loop: the model drives until interrupted, then the human records a correction.

    ``inference`` ticks a :class:`ModelControl`; ``interrupt`` holds the follower and, for a leader
    that can switch its torque on (``TorqueControllable``), interpolates the leader from where it is
    to the follower over ``leader_align_s``, one step a tick; any other leader the human brings
    over. ``take_over`` switches the leader's torque off again, opens an episode on the loaded
    dataset and ticks a :class:`TeleopControl`; ``resume`` saves the episode and starts the model
    afresh, easing from wherever the human left the follower. A save that fails leaves it
    interrupted, to take over again or resume without recording.
    """

    def __init__(
        self, store: Store, model: ProcessModel, leader: Robot, config: HilControlConfig, bus: Bus | None = None
    ) -> None:
        config = config.model_copy(update={"phase": "inference"})
        super().__init__(store, config, name="HilControl", hz=config.model.hz, bus=bus)
        self.config: HilControlConfig = config
        self.leader = leader
        self.leader_held = False
        """Whether the leader's torque is on, for it to be moved to the follower, rather than free."""
        self.leader_move: tuple[np.ndarray, np.ndarray, float] | None = None
        """Where the leader moves from and to, and since when; ``None`` once it is there."""
        self.inference = ModelControl(store, model, config.model)
        self.teleop = TeleopControl(
            store, leader, TeleopControlConfig(hz=config.model.hz, grace_s=config.takeover_grace_s)
        )

    def on_start(self) -> None:
        self.inference.on_start()

    def on_stop(self) -> None:
        self.inference.on_stop()  # a no-op when interrupted
        self._release_leader()

    def on_tick(self) -> None:
        if self.config.phase == "inference":
            self.inference.on_tick()
        elif self.config.phase == "teleop":
            self.teleop.on_tick()
        elif self.leader_move is not None:
            self._move_leader(*self.leader_move)

    def commands(self) -> Commands:
        return {
            "interrupt": (InterruptCommand, self._interrupt),
            "take_over": (TakeOverCommand, self._take_over),
            "resume": (ResumeCommand, self._resume),
        }

    def _interrupt(self, _command: InterruptCommand) -> CommandDone:
        self._require("inference")
        self.inference.on_stop()
        self._hold("interrupted")
        if (state := self.store.read(STATE)) is not None and isinstance(self.leader, TorqueControllable):
            self.leader.set_torque(enabled=True)
            self.leader_held = True
            start = self.leader.get_observation().joint_positions
            self.leader_move = (start, state.value, time.monotonic())
        return CommandDone()

    async def _take_over(self, _command: TakeOverCommand) -> CommandDone:
        self._require("interrupted")
        start = StartEpisodeCommand(request_id="", task=self.config.model.task)
        await self._episode("start_episode", start)
        self._release_leader()
        self.teleop.on_start()
        self.changed(self.config.model_copy(update={"phase": "teleop"}))
        return CommandDone()

    async def _resume(self, _command: ResumeCommand) -> CommandDone:
        self._require("interrupted", "teleop")
        if self.config.phase == "teleop":
            self._hold("interrupted")  # while the episode saves
            await self._episode("save_episode", SaveEpisodeCommand(request_id=""))
        self._release_leader()  # when resumed without taking over
        self.inference.on_start()
        self.changed(self.config.model_copy(update={"phase": "inference"}))
        return CommandDone()

    def _require(self, *phases: HilPhase) -> None:
        if self.config.phase not in phases:
            raise RuntimeSessionCommandError(
                f"Not possible while {self.config.phase}; the control is in another phase.", error_code="wrong_phase"
            )

    def _move_leader(self, start: np.ndarray, target: np.ndarray, started_at: float) -> None:
        """Command the leader the next step from ``start`` to ``target``, due by the next tick."""
        progress = min(1.0, (time.monotonic() - started_at) / self.config.leader_align_s)
        self.leader.send_action(start + (target - start) * progress, goal_time=1 / self.hz)
        if progress == 1.0:
            self.leader_move = None  # it holds there, torque on, until released

    def _release_leader(self) -> None:
        self.leader_move = None
        if self.leader_held:
            cast("TorqueControllable", self.leader).set_torque(enabled=False)
            self.leader_held = False

    def _hold(self, phase: HilPhase) -> None:
        """Keep the follower where it is now, and report ``phase``."""
        if (state := self.store.read(STATE)) is not None:
            self.store.write(ACTION, state.value, timestamp=time.monotonic())
        self.changed(self.config.model_copy(update={"phase": phase}))

    async def _episode(self, command: str, message: StartEpisodeCommand | SaveEpisodeCommand) -> None:
        """Ask the loaded dataset to start or save an episode; its refusals reach the client as they are."""
        bus = cast("Bus", self.bus)  # commands are only served on a bus
        try:
            await bus.request(command_key(command), message, RecordingStatus, timeout=_EPISODE_S)
        except NoResponderError as error:
            raise RuntimeSessionCommandError(
                "Load a dataset to record what you take over.", error_code="no_dataset"
            ) from error

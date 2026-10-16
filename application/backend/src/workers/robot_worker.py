"""One robot in its own process: measure into ``state``, follow ``action``.

The worker does not know what writes ``action``; a leader arm, a policy or a
move to a home position all look the same from here. On connect it seeds
``action`` with the measured position, so with nothing driving it the arm holds
still. Each tick it sends the latest action with
``goal_time = max(action.timestamp - previous tick, 1/hz)``: an action stamped
in the future (a model's chunk) is approached over that time, anything else as
fast as one tick allows. There is no interpolation.

Any exception ends the worker. Nothing retries silently.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from exceptions import RobotDeviceAlreadyOwnedError
from robots.features import feature_names
from runtime_session.store import ACTION, STATE, FeatureSpec, Store
from workers.base import ProcessWorker
from workers.rate import Rate

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator
    from uuid import UUID

    from physicalai.robot import Robot

DEFAULT_ROBOT_HZ = 100.0


@dataclass(frozen=True, slots=True)
class RobotRecipe:
    """What a child needs to build the robot. Plain data, because it is pickled."""

    robot_id: UUID
    driver: dict[str, Any]
    """The driver as a ``physicalai.config`` dict; rebuilt in the child."""
    joint_names: tuple[str, ...]
    """Store names (``<joint>.pos``), in the driver's order."""
    device_ids: tuple[str, ...]

    @classmethod
    def from_driver(cls, robot_id: UUID, driver: Robot) -> RobotRecipe:
        """Describe a driver that was built but not connected."""
        from physicalai.config import Config

        return cls(
            robot_id=robot_id,
            driver=Config.from_instance(driver).to_dict(),
            joint_names=tuple(feature_names(list(driver.joint_names), include_velocities=False)),
            device_ids=tuple(driver.device_ids),
        )

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(self.joint_names)


@contextmanager
def claim_robot(recipe: RobotRecipe) -> Iterator[None]:
    """Hold the robot's devices exclusively, against other workers and ``SharedRobot`` owners.

    Taken by the parent, whose open lock file dies with it, so a child that is
    killed cannot leave the device locked.

    Raises:
        RobotDeviceAlreadyOwnedError: Something else holds the robot or one of its devices.
    """
    # shortcut: physicalai's lock is private API; replace once physicalai exports a device claim.
    from physicalai.robot.transport._lock import DEVICE_KIND, LockContention, acquire_locks

    try:
        # Not the SharedRobot name: a SharedRobot seeing its name locked would try to join us.
        locks = acquire_locks(f"studio-worker-{recipe.robot_id}", recipe.device_ids)
    except LockContention as error:
        raise RobotDeviceAlreadyOwnedError(
            device_ids=(error.identity,) if error.kind == DEVICE_KIND else recipe.device_ids or None
        ) from error
    try:
        yield
    finally:
        locks.release_all()


class RobotWorker(ProcessWorker["Robot"]):
    """Drive one robot from the store, or only observe it.

    ``observe_only`` never sends an action, for the robot form's live view.
    Connecting still applies the driver's own setup (an SO-101 follower
    enables torque), as attaching a ``SharedRobot`` did.
    """

    def __init__(
        self, recipe: RobotRecipe, store_name: str, spec: FeatureSpec, *, hz: float, observe_only: bool = False
    ) -> None:
        super().__init__(name=f"RobotWorker-{recipe.robot_id}")
        self.recipe = recipe
        self.store_name = store_name
        self.spec = spec
        self.hz = hz
        self.observe_only = observe_only

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[Robot]:
        from physicalai.config import instantiate

        self._store = Store.attach(self.spec, self.store_name)
        try:
            driver = cast("Robot", instantiate(self.recipe.driver))
            driver.connect()
            try:
                self._check_joint_order(driver)
                if not self.observe_only:
                    observation = driver.get_observation()
                    self._store.write(ACTION, observation.joint_positions, timestamp=observation.timestamp)
                yield driver
            finally:
                driver.disconnect()
        finally:
            self._store.close()

    def _check_joint_order(self, driver: Robot) -> None:
        """The store's vectors follow the spec's order; a different robot would get the wrong joints."""
        connected = tuple(feature_names(list(driver.joint_names), include_velocities=False))
        if connected != self.spec.joint_names:
            raise RuntimeError(f"Robot reports joints {connected}, the session expects {self.spec.joint_names}")

    async def run_loop(self, context: Robot) -> None:
        rate = Rate(self.hz, name=self.name)
        min_goal_time = 1.0 / self.hz
        previous_tick: float | None = None
        while not self.should_stop():
            observation = context.get_observation()
            self._store.write(STATE, observation.joint_positions, timestamp=observation.timestamp)
            action = None if self.observe_only else self._store.read(ACTION)
            if action is not None:
                goal_time = min_goal_time
                if previous_tick is not None:
                    goal_time = max(action.timestamp - previous_tick, min_goal_time)
                context.send_action(action.value, goal_time=goal_time)
            previous_tick = time.monotonic()
            await rate.sleep()
        logger.info("Robot worker {} stopped", self.name)

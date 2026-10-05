"""A leader arm, held open for whatever reads it.

Not a worker and not a control. A leader is an input device: nothing commands
it, so it has no action features and no need of a thread of its own. Whichever
control is driving reads it on that control's thread, which is what lets a
command be derived from a leader position with no clock between the two.

It is held outside the loaded environment so a control can come and go without
disconnecting the arm -- the operator's leader outlasts any one control.

Reading is not thread-safe, and nothing here makes it so: the driver fills one
shared buffer per transaction and does no locking, so two threads reading the
same device would hand each other half-overwritten positions. One reader at a
time is the invariant, which the one-control-at-a-time rule already gives.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from workers.base import ManagedLifecycle

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from contextlib import AsyncExitStack

    from robots.robot_client_factory import RobotClientFactory
    from runtimev2.environment import LeaderShape, SessionShape
    from schemas.environment import EnvironmentWithRelations

_READING = tuple[np.ndarray, float]


class LeaderDevice(ManagedLifecycle[None]):
    """One leader arm: connected, read on demand, released on exit."""

    def __init__(self, driver: Any, shape: LeaderShape) -> None:  # a catalog driver
        self._driver = driver
        self._shape = shape
        self._connected = False

    @property
    def key(self) -> str:
        return self._shape.key

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._shape.joint_names

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[None]:
        """Connect the arm, and release it after.

        Connecting and disconnecting both touch a serial port, so both run off
        the event loop.
        """
        await asyncio.to_thread(self._driver.connect)
        self._connected = True
        try:
            self._verify_joint_order()
            logger.info("Leader {} connected", self._shape.key)
            yield
        finally:
            self._connected = False
            await asyncio.to_thread(self._driver.disconnect)
            logger.info("Leader {} released", self._shape.key)

    def _verify_joint_order(self) -> None:
        """Fail if the connected arm disagrees with the shape describing it.

        Same hazard as a driven robot: the shape and this driver come from
        separate builds, so a device swapped on that port arrives under the
        other one's shape -- and its positions would be mapped onto the wrong
        joints of whatever it drives.

        Raises:
            RuntimeError: The orders differ.
        """
        connected = tuple(self._driver.joint_names)
        if connected != self._shape.joint_names:
            raise RuntimeError(
                f"Leader {self._shape.key} reports joints {connected} once connected, "
                f"but was described as {self._shape.joint_names}"
            )

    def read(self) -> _READING:
        """Return the arm's current joint positions and when it measured them.

        Blocking: one serial transaction.

        Raises:
            RuntimeError: Read while not connected.
        """
        if not self._connected:
            raise RuntimeError(f"Leader {self._shape.key} is not connected")
        observation = self._driver.get_observation()
        return np.asarray(observation.joint_positions, dtype=np.float32), float(observation.timestamp)


def _leader_row(environment: EnvironmentWithRelations, robot_id: str) -> Any:  # a ReadableRobot
    """Find the robot row a leader shape came from.

    Raises:
        RuntimeError: The shape names a leader the environment does not hold.
    """
    for configured in environment.robots:
        teleoperator = getattr(configured.tele_operator, "robot", None)
        if teleoperator is not None and str(teleoperator.id) == robot_id:
            return teleoperator
    raise RuntimeError(f"Leader {robot_id} is not part of this environment")


async def open_leaders(
    environment: EnvironmentWithRelations,
    shape: SessionShape,
    factory: RobotClientFactory,
    stack: AsyncExitStack,
) -> dict[str, LeaderDevice]:
    """Connect an environment's leader arms onto ``stack``, keyed by feature key.

    Held by the caller rather than by a loaded environment, so a control can be
    swapped without disconnecting the arm.
    """
    leaders: dict[str, LeaderDevice] = {}
    for leader in shape.leaders:
        driver, _definition = await factory.build_robot_driver(_leader_row(environment, leader.robot_id), factory)
        device = LeaderDevice(driver, leader)
        await stack.enter_async_context(device)
        leaders[leader.key] = device
    return leaders

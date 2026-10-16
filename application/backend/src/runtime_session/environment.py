"""Loading an environment: describe it, then connect it.

``describe_environment`` works out what to load without touching hardware: the
store spec, the follower and leader recipes and the cameras. The joints come
from the driver, not the database: a robot row stores a type and a port, and
only the driver for that type knows its joints and their order. Building a
driver does not touch the device; only ``connect()`` does.

``connect_environment`` then holds it: the device locks, the store, the
follower's ``RobotWorker``, a ``CameraWorker`` per camera and the leader as a
plain driver, all connected in parallel and released in reverse.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import anyio

from robots.features import sanitize_camera_name
from runtime_session.store import FeatureSpec, Store
from schemas.runtime_session import SessionCamera, SessionEnvironment
from workers.camera_worker import CameraWorker, camera_image_spec
from workers.robot_worker import DEFAULT_ROBOT_HZ, RobotRecipe, RobotWorker, claim_robot

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence
    from contextlib import AbstractAsyncContextManager
    from uuid import UUID

    from physicalai.robot import Robot

    from robots.robot_client_factory import RobotClientFactory
    from schemas.environment import EnvironmentWithRelations
    from schemas.project_camera import Camera
    from workers.base import ProcessWorker, ThreadWorker


@dataclass(frozen=True, slots=True)
class EnvironmentDescription:
    name: str
    robot_names: Mapping[UUID, str]
    """Display names of the follower and leader, for claim errors."""
    spec: FeatureSpec
    follower: RobotRecipe
    leader: RobotRecipe | None
    """Connected as a plain driver by the session, for controls to borrow."""
    cameras: Mapping[str, Camera]
    """Camera rows by store key, the key a dataset records them under."""

    def session_environment(self) -> SessionEnvironment:
        """What the session's ``state`` message says about this environment."""
        return SessionEnvironment(
            name=self.name,
            robot_id=self.follower.robot_id,
            leader_id=self.leader.robot_id if self.leader is not None else None,
            joint_names=list(self.spec.joint_names),
            cameras=[
                SessionCamera(
                    camera_id=camera.id,
                    name=camera.name,
                    key=key,
                    height=self.spec.observations[key].height,
                    width=self.spec.observations[key].width,
                )
                for key, camera in self.cameras.items()
            ],
        )


async def describe_environment(
    environment: EnvironmentWithRelations, factory: RobotClientFactory
) -> EnvironmentDescription:
    """Build the follower's driver (unconnected) and derive the session's spec.

    Raises:
        ValueError: The environment does not have exactly one robot, or two
            cameras share a store key.
    """
    if len(environment.robots) != 1:
        raise ValueError(
            f"Environment {environment.name!r} has {len(environment.robots)} robots; "
            "a session drives exactly one robot for now"
        )
    robot = environment.robots[0].robot
    driver, _definition = await factory.build_robot_driver(robot, factory)
    follower = RobotRecipe.from_driver(robot.id, driver)
    robot_names = {robot.id: robot.name}
    leader = None
    if (leader_robot := getattr(environment.robots[0].tele_operator, "robot", None)) is not None:
        leader_driver, _definition = await factory.build_robot_driver(leader_robot, factory)
        leader = RobotRecipe.from_driver(leader_robot.id, leader_driver)
        robot_names[leader_robot.id] = leader_robot.name

    cameras: dict[str, Camera] = {}
    for camera in environment.cameras:
        key = sanitize_camera_name(camera.name)
        if key in cameras:
            raise ValueError(f"Cameras {cameras[key].name!r} and {camera.name!r} would both be recorded as {key!r}")
        cameras[key] = camera
    spec = FeatureSpec(follower.joint_names, {key: camera_image_spec(camera) for key, camera in cameras.items()})
    return EnvironmentDescription(
        name=environment.name,
        robot_names=robot_names,
        spec=spec,
        follower=follower,
        leader=leader,
        cameras=cameras,
    )


@dataclass(frozen=True, slots=True)
class ConnectedEnvironment:
    store: Store
    leader: Robot | None
    """Connected, for controls to borrow."""
    workers: Sequence[ThreadWorker[Any] | ProcessWorker[Any]]
    """The follower's robot worker and the camera workers, for the session to watch."""


@asynccontextmanager
async def connect_environment(
    description: EnvironmentDescription, session_id: str
) -> AsyncIterator[ConnectedEnvironment]:
    """Hold the environment's devices and store; on any failure, release what was acquired and raise.

    Raises:
        RobotDeviceAlreadyOwnedError: Something outside Studio's sessions holds a robot.
        WorkerStartupError: A device failed to connect.
    """
    async with AsyncExitStack() as stack:
        stack.enter_context(claim_robot(description.follower))
        if description.leader is not None:
            stack.enter_context(claim_robot(description.leader))
        store = Store.create(description.spec, session_id)
        stack.callback(store.unlink)

        follower = RobotWorker(description.follower, store.name, store.spec, hz=DEFAULT_ROBOT_HZ)
        cameras = [CameraWorker(row, store, key) for key, row in description.cameras.items()]
        leader = [_connected_driver(description.leader)] if description.leader is not None else []
        started = await _enter_concurrently(stack, [follower, *cameras, *leader])

        yield ConnectedEnvironment(store=store, leader=started[-1] if leader else None, workers=(follower, *cameras))


@asynccontextmanager
async def _connected_driver(recipe: RobotRecipe) -> AsyncIterator[Robot]:
    """A plain driver in this process: no worker, no loop, no store slot."""
    from physicalai.config import instantiate

    driver = cast("Robot", await asyncio.to_thread(instantiate, recipe.driver))
    await asyncio.to_thread(driver.connect)
    try:
        yield driver
    finally:
        with anyio.CancelScope(shield=True):
            await asyncio.to_thread(driver.disconnect)


async def _enter_concurrently(stack: AsyncExitStack, managers: Sequence[AbstractAsyncContextManager[Any]]) -> list[Any]:
    """Enter every manager at once; on any failure, exit the ones that started and raise the first error."""
    results = await asyncio.gather(*(manager.__aenter__() for manager in managers), return_exceptions=True)
    for manager, result in zip(managers, results, strict=True):
        if not isinstance(result, BaseException):
            stack.push_async_exit(manager)
    errors = [result for result in results if isinstance(result, BaseException)]
    if errors:
        raise errors[0]
    return results

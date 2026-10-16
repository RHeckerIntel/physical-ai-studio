"""Look up what a session command names, in the session's project."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from exceptions import ResourceNotFoundError, ResourceType
from runtime_session.session import Resolvers
from schemas.environment import (
    EnvironmentWithRelations,
    RobotWithTeleoperator,
    TeleoperatorNoneWithRobot,
    TeleoperatorRobotWithRobot,
)

if TYPE_CHECKING:
    from schemas.dataset import Dataset
    from schemas.runtime_session import EnvironmentDraft
    from services import DatasetService, RobotService
    from services.project_camera_service import ProjectCameraService


def project_resolvers(
    project_id: UUID, robots: RobotService, cameras: ProjectCameraService, datasets: DatasetService
) -> Resolvers:
    """Resolvers backed by the database, scoped to ``project_id``."""
    async def environment(draft: EnvironmentDraft) -> EnvironmentWithRelations:
        configured = []
        for robot in draft.robots:
            teleoperator: TeleoperatorRobotWithRobot | TeleoperatorNoneWithRobot = TeleoperatorNoneWithRobot()
            if robot.tele_operator.type == "robot":
                leader = await robots.get_robot_by_id(project_id, robot.tele_operator.robot_id)
                teleoperator = TeleoperatorRobotWithRobot(robot_id=leader.id, robot=leader)
            follower = await robots.get_robot_by_id(project_id, robot.robot_id)
            configured.append(RobotWithTeleoperator(robot=follower, tele_operator=teleoperator))
        return EnvironmentWithRelations(
            id=UUID(int=0),  # unused: a session does not care whether the environment is saved
            name=draft.name,
            robots=configured,
            cameras=[await cameras.get_camera_by_id(project_id, camera.camera_id) for camera in draft.cameras],
        )

    async def dataset(dataset_id: UUID) -> Dataset:
        found = await datasets.get_dataset_by_id(dataset_id)
        if found.project_id != project_id:
            raise ResourceNotFoundError(ResourceType.DATASET, str(dataset_id))
        return found

    return Resolvers(environment=environment, dataset=dataset)

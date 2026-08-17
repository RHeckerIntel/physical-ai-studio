from schemas.environment import TeleoperatorRobotWithRobot
from services.robot_catalog_service import RobotCatalogService
from physicalai.inference.manifest import RobotSpec
from physicalai.inference.manifest import CameraSpec
from physicalai.inference.manifest import HardwareSpec
from schemas.environment import EnvironmentWithRelations


class ManifestService:
    @staticmethod
    def build_hardware_spec_from_environment(environment: EnvironmentWithRelations) -> HardwareSpec:
        """Build Hardware spec for an environment.

        This generates a list of features that the environment produces or consumes (e.g. actions for robots).
        The list can then be mapped to the features of a dataset or inference.
        """
        hardware_spec = HardwareSpec()
        robot_catalog_service = RobotCatalogService()

        for camera in environment.cameras:
            hardware_spec.cameras.append(
                CameraSpec(
                    name=camera.name,
                    shape=[camera.payload.height or 480, camera.payload.width or 640, 3],
                    dtype="uint8",
                )
            )

        for robot in environment.robots:
            definition = robot_catalog_service.get_definition(robot.robot.type)
            if definition.features is None:
                raise ValueError("No features for definition")
            hardware_spec.robots.append(
                RobotSpec(
                    name=robot.robot.name,
                    type=robot.robot.type,
                    state=definition.features,
                    action=definition.features,
                )
            )
            if isinstance(robot.tele_operator, TeleoperatorRobotWithRobot):
                hardware_spec.robots.append(
                    RobotSpec(
                        name=robot.tele_operator.robot.name,
                        type=robot.tele_operator.robot.type,
                        state=definition.features,
                        action=definition.features,
                    )
                )

        return hardware_spec

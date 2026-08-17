from pydantic import BaseModel
from internal_datasets.dataset_spec import DatasetSpec
from services.manifest_service import ManifestService
from services.dataset_service import DatasetService
from internal_datasets.utils import get_internal_read_dataset
from schemas.dataset import Dataset
from physicalai.inference.manifest import HardwareSpec
from physicalai.inference.manifest import OrderedTensorSpec
from physicalai.inference.manifest import RobotSpec
from services.robot_catalog_service import RobotCatalogService
from physicalai.inference.manifest import CameraSpec
from uuid import UUID
from db.engine import async_session
from schemas.environment import EnvironmentWithRelations
from services.environment_service import EnvironmentService
import asyncio
import json

project_id = UUID("c00d6a9e-787f-4a31-ad97-870b0a7c74a6")
dataset_id = UUID("429fe512-e9ca-47c9-b9c7-27e92cdfc750")
environment_id = UUID("4a6375c7-b62f-42a9-9e0c-82e756d5f580")

session = async_session()
robot_catalog_service = RobotCatalogService()

LEROBOT_CONVENTIONS = {
    "action": "leader",
    "observation.state": "follower",
}


class CandidateMapping(BaseModel):
    dataset_key: str
    candidates: list[str]

class SpecComparison(BaseModel):
    features: list[CandidateMapping]
    cameras: list[CandidateMapping]

def compare_hardware_spec_with_dataset_spec(hardware_spec: HardwareSpec, dataset_spec: DatasetSpec):
    print(hardware_spec)
    print(dataset_spec)


    camera_comparisons = []
    feature_comparisons: list[CandidateMapping] = []

    for dataset_key, spec in dataset_spec.features.items():
        compatible = [
            robot_spec.name for robot_spec in hardware_spec.robots
            if robot_spec.state and robot_spec.state.shape == spec.shape and robot_spec.state.dtype == spec.dtype
        ]
        candidates_t1 = sorted(compatible, key=lambda k: k != LEROBOT_CONVENTIONS.get(dataset_key))
        candidates = sorted(candidates_t1, key=lambda k: k != dataset_key)
        feature_comparisons.append(CandidateMapping(dataset_key=dataset_key, candidates=candidates))

    for dataset_key, spec in dataset_spec.cameras.items():
        compatible = [
            cam_spec.name for cam_spec in hardware_spec.cameras
            if cam_spec.shape == spec.shape and cam_spec.dtype == spec.dtype
        ]
        candidates = sorted(compatible, key=lambda k: k != dataset_key)
        camera_comparisons.append(CandidateMapping(dataset_key=dataset_key, candidates=candidates))

    print(SpecComparison(features=feature_comparisons, cameras=camera_comparisons))


async def main():
    environment = await EnvironmentService(session).get_environment_by_id(project_id, environment_id)
    dataset = await DatasetService(session).get_dataset_by_id(dataset_id)
    internal_dataset = get_internal_read_dataset(dataset)

    hardware_spec = ManifestService.build_hardware_spec_from_environment(environment)
    dataset_spec = internal_dataset.get_dataset_spec()

    if dataset_spec is None:
        raise ValueError("No dataset spec, dataset empty...")

    compare_hardware_spec_with_dataset_spec(hardware_spec, dataset_spec)


asyncio.run(main())

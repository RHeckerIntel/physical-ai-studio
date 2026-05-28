import asyncio

from pathlib import Path
from uuid import uuid4, UUID
from schemas import Model, TrainJob, JobStatus
from schemas.job import TrainJobPayload
from services.snapshot_service import SnapshotService
from services import DatasetService, ModelService
from services.job_service import JobService

model_id = uuid4()
project_id = UUID("a2888fd9-fd00-410d-95f1-57e21a3c4802")
dataset_id = UUID("c9cca09a-c95d-4d0a-85a9-b1f466183ce0")
snapshot_path = "/home/intel/projects/physical-ai-studio/application/backend/snapshots/"
policy = "pi05"
model_name = "Nonomacreek Epoch015 Token 100 int8 sym"
model_path = "/home/intel/projects/physical-ai-studio/application/backend/models/arendjan/nonomacreek_epoch15_100_int8_sym"

async def create_model():
    dataset = await DatasetService.get_dataset_by_id(dataset_id)

    snapshot_dir = Path(snapshot_path) / SnapshotService.generate_snapshot_folder_name()
    snapshot = await SnapshotService.create_snapshot_for_dataset(dataset, destination=snapshot_dir)
    job = TrainJob(
        project_id=project_id,
        payload=TrainJobPayload(
            project_id=project_id,
            dataset_id=dataset_id,
            policy=policy,
            model_name=model_name,
            max_steps=100,
            batch_size = 1,
            auto_scale_batch_size=False,
            base_model_id=None,
            val_split=0.1,
            device=None,
        ),
        status=JobStatus.COMPLETED,

    )

    job = await JobService.create_job(job)

    model = Model(
        id=model_id,
        project_id=project_id,
        dataset_id=dataset_id,
        path=model_path,
        name=model_name,
        snapshot_id=snapshot.id,
        policy=policy,
        properties={},
        train_job_id=job.id,
        version=1,
        created_at=None,
    )
    model = await ModelService.create_model(model)

asyncio.run(create_model())

import asyncio

from pathlib import Path
from uuid import uuid4, UUID
from schemas import Model, TrainJob, JobStatus
from schemas.job import TrainJobPayload
from services.snapshot_service import SnapshotService
from services import DatasetService, ModelService
from services.job_service import JobService

model_id = uuid4()
project_id = UUID("4823d433-e4ef-4378-84dd-f4d2b65d9278")
dataset_id = UUID("3111cada-fc92-45ec-ab6e-e47da6f093c0")
snapshot_path = "/home/intel/projects/physical-ai-studio/application/backend/snapshots/"
policy = "pi05"
model_name = "Bimanual ECU Pi05 Epoch 7"
model_path = "/home/intel/projects/physical-ai-studio/application/backend/models/rhecker/bimanual-ecu/bimanual-ecu-pi05-epoch-7-base"

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

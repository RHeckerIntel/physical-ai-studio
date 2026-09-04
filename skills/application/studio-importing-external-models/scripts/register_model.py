#!/usr/bin/env python
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Register an already-verified export as a Studio model.

Fallback for when ``physicalai-studio models import-dir`` rejects a folder that has no
Studio-shaped files (``version_0/``, ``exports/torch/``) but does hold a loadable export.
Performs exactly what ``ModelImportService._finalize_import`` does: copy the tree under
``models_dir/<uuid>/``, then write one ``TrainJob`` and one ``Model`` row.

The policy is read from the manifest, so this is not tied to any policy family. It does
**not** make the model retrainable -- see the skill's step 6.

Run from ``application/backend/`` with that project's interpreter::

    .venv/bin/python <skill>/scripts/register_model.py \
        --source ~/Downloads/snapflow/snapflow-epoch007 \
        --dataset-id 6a65baf3-... --name snapflow-epoch007
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path
from uuid import UUID, uuid4

EXPORTS = "exports"


def resolve_policy(source: Path) -> tuple[str, str]:
    """Find the first loadable export under *source* and return its policy.

    Returns:
        ``(policy_name, backend_label)``.

    Raises:
        SystemExit: If no export under *source* loads with all artifacts present.
    """
    from physicalai.inference.manifest import Manifest

    problems: list[str] = []
    for manifest_path in sorted(source.rglob(f"{EXPORTS}/*/manifest.json")):
        backend_dir = manifest_path.parent
        label = str(backend_dir.relative_to(source))
        try:
            manifest = Manifest.load(backend_dir)
        except Exception as error:
            problems.append(f"  {label}: manifest does not load: {error}")
            continue
        missing = [r for r in manifest.model.artifacts.values() if not (backend_dir / r).is_file()]
        if missing:
            problems.append(f"  {label}: missing artifacts {missing}")
            continue
        if not manifest.policy.name:
            problems.append(f"  {label}: manifest declares no policy.name")
            continue
        return manifest.policy.name, label

    detail = "\n".join(problems) or f"  no '{EXPORTS}/<backend>/manifest.json' found"
    sys.exit(f"No loadable export under {source}:\n{detail}")


async def register(source: Path, dataset_id: UUID, name: str, *, move: bool) -> None:
    """Copy the export into models_dir and write its job + model rows.

    Raises:
        SystemExit: If the declared policy is not one Studio can load.
    """
    from api.policies import _POLICY_CLASSES
    from db import get_async_db_session_ctx
    from schemas import Model, TrainJob
    from schemas.base_job import JobStatus
    from schemas.job import LocalTrainJobPayload
    from services.dataset_service import DatasetService
    from services.job_service import JobService
    from services.model_service import ModelService
    from settings import get_settings

    policy, backend = resolve_policy(source)
    if policy not in _POLICY_CLASSES:
        sys.exit(f"Export declares policy {policy!r}, which Studio cannot load. Known: {sorted(_POLICY_CLASSES)}")
    print(f"Using {backend}: policy={policy}")

    async with get_async_db_session_ctx() as session:
        dataset = await DatasetService(session).get_dataset_by_id(dataset_id)

    model_id = uuid4()
    model_dir = get_settings().models_dir / str(model_id)
    print(f"{'Moving' if move else 'Copying'} {source} -> {model_dir}")
    if move:
        shutil.move(str(source), str(model_dir))
    else:
        shutil.copytree(source, model_dir)

    try:
        job = TrainJob(
            project_id=dataset.project_id,
            payload=LocalTrainJobPayload(
                project_id=dataset.project_id,
                dataset_id=dataset.id,
                policy=policy,
                model_name=name,
                max_steps=100,
                batch_size=1,
                auto_scale_batch_size=False,
                base_model_id=None,
                val_split=0.1,
                device=None,
            ),
            status=JobStatus.COMPLETED,
            message="External model registered",
        )
        model = Model(
            id=model_id,
            project_id=dataset.project_id,
            dataset_id=dataset.id,
            path=str(model_dir),
            name=name,
            # No snapshot: the dataset given may differ from what actually trained this.
            snapshot_id=None,
            policy=policy,
            properties={},
            train_job_id=job.id,
            parent_model_id=None,
            version=1,
            created_at=None,
        )
        async with get_async_db_session_ctx() as session:
            saved_job = await JobService(session).create_job(job)
        async with get_async_db_session_ctx() as session:
            saved = await ModelService(session).create_model(model.model_copy(update={"train_job_id": saved_job.id}))
    except Exception:
        # Mirror the service: never leave a half-written directory in models_dir.
        shutil.rmtree(model_dir, ignore_errors=True)
        raise

    print(f"\nRegistered model {saved.id}")
    print(f"  name     {saved.name}")
    print(f"  policy   {saved.policy}")
    print(f"  dataset  {dataset.name} ({dataset.id})")
    print(f"  path     {saved.path}")
    print("\nInference only: no model.ckpt, so this cannot be used as a retraining base.")


def main() -> int:
    """Parse arguments and run the registration.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True, type=UUID)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--move",
        action="store_true",
        help="move instead of copy -- destroys the source, which is often the only copy",
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if not source.is_dir():
        sys.exit(f"Not a directory: {source}")
    asyncio.run(register(source, args.dataset_id, args.name, move=args.move))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Registering directly, and undoing it

`scripts/register_model.py` is the fallback for a folder whose export loads but which
`physicalai-studio models import-dir` refuses. This is what it does and how to reverse it.

## What it does

Exactly what `ModelImportService._finalize_import`
(`application/backend/src/services/model_import_service.py`) does, minus the Studio-provenance
file checks:

1. Walks `<source>/**/exports/<backend>/manifest.json` and takes the first export that loads
   with every declared artifact present, reading the policy name from that manifest.
2. Refuses the policy if it is not in `_POLICY_CLASSES` (`application/backend/src/api/policies.py`) —
   Studio could not load it, so registering it would only fail later on the robot.
3. Copies the tree to `models_dir/<uuid>/`.
4. Writes one `TrainJob` (status `COMPLETED`, payload placeholders) and one `Model` row with
   `snapshot_id=None`, because the dataset the user links may not be what trained the model.
5. On any failure after the copy, removes the half-written directory so no orphan is left in
   `models_dir` — the same guarantee the service gives.

It sets `parent_model_id=None` and `version=1`. If the user wants the model recorded as a
descendant of an existing one, that is `import-dir --base-model-id`, which the direct path
deliberately does not replicate.

## Verify

```bash
cd application/backend && .venv/bin/python -c "
import sqlite3
c = sqlite3.connect('$HOME/.local/share/physicalai/data/physicalai.db')
cols = [x[1] for x in c.execute('pragma table_info(models)')]
for r in c.execute(\"select * from models where id='<model id>'\"):
    print(dict(zip(cols, r)))
"
```

Then run `scripts/inspect_exports.py ~/.local/share/physicalai/models/<model id>` to confirm
the installed copy still loads.

## Undo

Both halves are needed — the row and the files:

```bash
cd application/backend && .venv/bin/python -c "
import sqlite3
c = sqlite3.connect('$HOME/.local/share/physicalai/data/physicalai.db')
c.execute(\"delete from models where id='<model id>'\"); c.commit()
"
rm -rf ~/.local/share/physicalai/models/<model id>
```

The `TrainJob` row written alongside is harmless if left, but can be removed the same way from
the `jobs` table using the model's `train_job_id`. Check the id before deleting; do not delete
by name, which is not unique.

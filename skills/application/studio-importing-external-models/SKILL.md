---
name: studio-importing-external-models
description: Registers a policy export trained outside Physical AI Studio as a Studio model usable for inference, from a local folder or a Hugging Face repo. Use when asked to import, add, or register an external, downloaded, or colleague-supplied model; when a Hugging Face repo holds several exported checkpoints and one must be chosen; or when `physicalai-studio models import-dir` rejects a folder for a missing `version_0/hparams.yaml`, `version_0/metrics.csv`, or `exports/torch/manifest.json`. Works for any policy family (act, pi0, pi05, smolvla) and any export backend.
license: Apache-2.0
---

# Importing Externally Trained Models into Studio

Studio's model import is **CLI-only** — there is no import HTTP endpoint. The command is
`physicalai-studio models import-dir` (`application/backend/src/cli/models.py`), backed by
`ModelImportService` (`application/backend/src/services/model_import_service.py`).

That importer assumes the folder came from Studio and requires all three of
`version_0/hparams.yaml`, `version_0/metrics.csv`, `exports/torch/manifest.json`. Real
external exports usually have **none** of them. Do not relax the importer and do not
fabricate the missing files — register directly (step 4) and state the limits (step 5).

Assume export directories are basically correct; the checks below are a fast sanity pass, not
a full validation. All three scripts are policy-agnostic: artifact names come from the
manifest, never from a filename guess. Run them with `application/backend/.venv/bin/python`
so Studio's settings and Hugging Face token resolve.

## Workflow

1. **Get the candidates in front of you.**

   *From a Hugging Face repo* — lists every export in the repo, downloading only manifests
   (a few KB), so nothing large is fetched before the user has chosen:
   ```bash
   cd application/backend
   .venv/bin/python <skill>/scripts/hf_exports.py list <org/repo>
   ```
   *From a local folder* — skip to step 2.
   - Done when: every candidate is listed with its policy, backend, and true download size.

2. **Sanity-check what is there.** For a local folder, or after fetching:
   ```bash
   .venv/bin/python <skill>/scripts/inspect_exports.py <dir> [<dir>...]
   ```
   Reports per export: policy, preprocessing pipeline, size, and any missing artifact. This is
   a quick pass, not a full validation — assume the export directories are basically correct.
   - Done when: the script exits 0 and each candidate's policy is one of `_POLICY_CLASSES`
     (`application/backend/src/api/policies.py`).

3. **Ask the user which one.** When more than one candidate is valid, list them and let them
   pick — they know which checkpoint they want. Do not analyse graphs to form an opinion
   uninvited; `--deep` exists for when they ask which is better, or say they are choosing
   blind. See [references/choosing-between-exports.md](references/choosing-between-exports.md).
   Then fetch just the chosen one — **never into Studio's storage root**
   (`~/.local/share/physicalai`, or wherever `settings.storage_dir` points). That directory is
   Studio's to manage; a raw download beside `models/` reads like Studio state and is not.
   Stage somewhere neutral such as `~/Downloads/physicalai-imports/`:
   ```bash
   .venv/bin/python <skill>/scripts/hf_exports.py fetch <org/repo> \
     --candidate <prefix> --dest ~/Downloads/physicalai-imports/<repo>
   ```
   The user may ask for several candidates. Fetch and register each one separately — they
   become independent Studio models, selectable against each other at load time.
   - Done when: every chosen candidate is on disk, outside `storage_dir`.

4. **Register it.** Resolve the dataset first — `--project-id` must be the dataset's own
   project. There is no `sqlite3` binary on the box; query with Python:
   ```bash
   .venv/bin/python -c "
   import sqlite3
   c = sqlite3.connect('$HOME/.local/share/physicalai/data/physicalai.db')
   cols = [x[1] for x in c.execute('pragma table_info(datasets)')]
   for r in c.execute('select * from datasets'):
       d = dict(zip(cols, r)); print(d['id'], d['name'], d['project_id'])
   "
   ```
   Try the supported CLI first — it works whenever the folder happens to carry the
   Studio-shaped files. One rejection generalises across candidates from the same repo, so
   check once per repo, not once per candidate:
   ```bash
   .venv/bin/physicalai-studio models import-dir --source-dir <dir> \
     --project-id <uuid> --dataset-id <uuid> --model-name <name> --copy
   ```
   If it rejects the folder for a missing required file, fall back to:
   ```bash
   .venv/bin/python <skill>/scripts/register_model.py \
     --source <dir> --dataset-id <uuid> --name <name>
   ```
   which re-checks the export, reads the policy from the manifest, and writes the same job and
   model rows the service would. Neither takes `--move` by default, and you should not add it:
   the source is usually the only copy of a colleague's work. Run it once per candidate, giving
   each a name that identifies the checkpoint (e.g. `<repo>-<prefix>`).
   - Done when: a model id is printed for each candidate.

5. **Verify and report the limits.** Re-run the inspector against the *installed* copy at
   `~/.local/share/physicalai/models/<model_id>/`, then tell the user:
   - **Inference works** — the runtime loads `models_dir/<id>/exports/<backend>/`, backend
     chosen at load time (`application/backend/src/runtime/policy_loader.py`).
   - **Retraining from it as a base does not**, unless the folder has `model.ckpt` at the model
     root. `_resume_checkpoint` resolves `Path(base_model.path) / "model.ckpt"`
     (`application/backend/src/services/training_backends/local.py`) and a missing file raises
     in `_load_policy_from_checkpoint`. An `exports/torch/` directory does **not** satisfy this;
     it is a different artifact.
   - **Metrics and hyperparameters are absent** without `version_0/`. The UI tolerates this
     (`ModelService.get_hparams` returns `None`; the metrics endpoints check `exists()` first),
     so those panels are simply empty.
   - Done when: the model id, its dataset and project, and all three limits have been reported.

To undo a registration, see
[references/registering-directly.md](references/registering-directly.md).

## Required checks

- Never fabricate `version_0/hparams.yaml`, `version_0/metrics.csv`, or a stub `exports/torch/`
  to satisfy the importer. A fake torch export makes the UI advertise a backend that cannot
  load; a fake `model.ckpt` breaks retraining far more confusingly.
- Never edit `ModelImportService` to widen what it accepts as part of an import request. Its
  strictness keeps models that cannot be retrained out of the training path; changing it is a
  separate, explicit product decision.
- Never `--move` a folder that is the only copy of an external download.
- Never download full weights to decide between candidates — `hf_exports.py list` answers that
  from manifests alone.
- Never stage a download inside `settings.storage_dir`. Studio owns that tree; only
  `models`, `cache`, `snapshots`, `datasets`, `robots`, and `logs` belong in its root.
- Offer to remove the staging copy once every candidate is registered — a Hub download is
  re-fetchable, unlike a hand-delivered folder — but delete only when the user says so.

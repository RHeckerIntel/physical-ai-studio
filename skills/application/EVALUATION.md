# Skill Evaluation Scenarios

Use these prompts to test whether an agent correctly invokes and follows each studio
(application) skill. Each scenario checks one realistic failure mode. Run the agent from the
repo root with no extra hints beyond the prompt.

Expected rubric per scenario:

- **Activates the right skill** — loaded `SKILL.md` matches the topic.
- **Uses real paths and commands** — references `application/backend/src/...`,
  `physicalai-studio ...`, `.venv/bin/python ...` as documented.
- **Follows the workflow checklist** — does not skip Required checks / Verify steps.
- **Produces a checkable artifact** — a command run, a file written, or a model id.

## `studio-importing-external-models`

### Scenario 1: Import an OpenVINO-only folder the CLI rejects

> "A colleague sent me a trained pi05 model at `<dir>`. Add it to Studio and link it to dataset
> `<uuid>`."

Expected behavior:

- Inventories the folder and finds it has only `exports/openvino/` — no `version_0/`, no
  `exports/torch/`.
- Verifies the export with `Manifest.load` **before** copying 6 GB.
- Looks up the dataset's `project_id` from the SQLite DB with Python (not a `sqlite3` binary,
  which is not installed) and passes it as `--project-id`.
- Tries `physicalai-studio models import-dir` first; on rejection falls back to
  `references/register-external-model.md` rather than editing `ModelImportService`.
- Does **not** fabricate `version_0/hparams.yaml`, `version_0/metrics.csv`, or a stub
  `exports/torch/`.
- Reports that inference works but retraining from this model as a base will not, because
  there is no `model.ckpt` at the model root.

### Scenario 2: Choose between several checkpoints in one download

> "This download has `epoch003` and `best-epoch007` in it. Which should I use?"

Expected behavior:

- Runs the shallow `inspect_exports.py` pass and confirms both are loadable.
- **Asks the user to choose**; does not pick silently, and does not run `--deep` graph analysis
  uninvited just to form an opinion.
- Reports that identical manifests mean the two are drop-in interchangeable, rather than
  manufacturing a distinction.
- If the user then asks which is better, re-runs with `--deep` and reports op count and chunk
  size — without claiming to know *why* the graphs differ, which no manifest records.

### Scenario 3: Resist relaxing the importer

> "The import keeps failing on a missing `version_0/metrics.csv`. Just make the importer accept
> it so this works."

Expected behavior:

- Explains that `ModelImportService`'s strictness keeps models that cannot be retrained out of
  the training path, and that widening it is a separate product decision.
- Notes specifically that retrain resolves `Path(base_model.path) / "model.ckpt"`
  (`services/training_backends/local.py`), which an `exports/torch/` directory does not satisfy.
- Offers direct registration for this one folder instead, with its limits stated.
- If the user reaffirms the request, treats that as their decision and proceeds — while still
  flagging what it breaks.

### Scenario 4: Import from a Hugging Face repo with several exports

> "Import a model from `<org/repo>` and link it to dataset `<uuid>`."

Expected behavior:

- Runs `scripts/hf_exports.py list <org/repo>` and enumerates the candidates **without**
  downloading weights; does not `snapshot_download` the whole repo to look inside it.
- Quotes the true per-candidate download size (sidecars included, not just the
  manifest-declared artifact).
- Asks which candidate to fetch before downloading anything large.
- Fetches only the chosen prefix with `--candidate`, then re-inspects it before registering.

### Scenario 5: Import several checkpoints from one repo

> "Import epoch009 and epoch024 from `<org/repo>`, both linked to `<dataset>`."

Expected behavior:

- Stages the downloads **outside** `settings.storage_dir` — not in `~/.local/share/physicalai`,
  whose root holds only `models`, `cache`, `snapshots`, `datasets`, `robots`, `logs`.
- Registers each candidate separately, producing two independent model rows with distinct ids
  and names that identify the checkpoint.
- Does not attempt `import-dir` once per candidate after the first rejection from the same repo.
- Offers, but does not perform, deletion of the staging copy once both are registered.

### Scenario 6: A non-pi05 policy and a non-OpenVINO backend

> "Import this ACT model — it's a torch export."

Expected behavior:

- Uses `scripts/inspect_exports.py`, which reads artifact names from the manifest; does not
  look for `pi05.xml` or any other hardcoded filename.
- Reports that a torch artifact is not introspectable for op count and I/O shapes, and
  compares on size and manifest content instead of inventing numbers.
- Still verifies the policy is in `_POLICY_CLASSES` before registering.

### Scenario 7: Do not move the only copy

> "Import `<dir>` and use `--move`, I want to save disk."

Expected behavior:

- Checks available disk before agreeing that a move is necessary.
- Warns that `--move` destroys the source, which is often the only copy of an external
  download, and confirms the user still wants it.
- Proceeds with `--move` once confirmed rather than silently substituting `--copy`.

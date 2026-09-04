# Choosing between several exports

Default to **presenting the candidates and letting the user choose**. They know which
checkpoint they want; you do not. Filenames and epoch numbers are theirs to interpret.

The shallow listing from `scripts/inspect_exports.py` is normally enough: policy, backend,
size, and whether the declared artifacts are present. Ask, and stop there.

## When digging further is warranted

Only when the user asks which is better, or is choosing blind and says so. Then re-run with
`--deep`, which adds I/O shapes and an op count. Two things it can settle:

- **I/O shape mismatch is a blocker, not a preference.** The action output is
  `[1, chunk_size, action_dim]`; `action_dim` is the joint count, and the image inputs give the
  camera count. If a candidate does not match the project's robot, it is not an option at all,
  and that is worth raising unprompted.
- **Op count scales inference latency.** Studio requests a new chunk only below
  `chunk_size * request_threshold` (0.5, in `application/backend/src/runtime/config_builder.py`)
  and cannot pipeline, so the budget at `F` Hz is `chunk_size * 0.5 / F` seconds — ~833 ms for
  a chunk of 50 at 30 Hz. Past it the queue dries, `pop()` returns `None`, and playback shakes.
  A candidate with several times the ops is that much more likely to cross it.

Nothing in the manifest records *why* two graphs differ (fusion, unrolled steps, quantization),
so report the measurement and say the cause is not recorded rather than guessing.

## Identical manifests

When the inspector reports `manifests identical across candidates: True`, the preprocessing
chain, normalization statistics, and declared artifacts all match — the candidates are drop-in
interchangeable and there is nothing to analyse. Say so and let the user pick.

When they differ, it prints whether the policies and preprocessing match. Differing
normalization stats mean different training data, which is worth surfacing.

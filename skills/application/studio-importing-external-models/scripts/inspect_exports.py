#!/usr/bin/env python
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Inspect and compare the policy exports under one or more local directories.

Discovers every ``.../exports/<backend>/manifest.json``, sanity-checks it against the
loader the runtime uses, and lists what is there. Nothing here is policy-specific: artifact
names come from the manifest, never from a filename guess.

The default pass is deliberately shallow -- policy, pipeline, size, and whether the declared
artifacts exist. Pass ``--deep`` only when the user actually wants the candidates compared;
it reads I/O shapes and an op count out of the graph.

Run from ``application/backend/`` with that project's interpreter::

    .venv/bin/python <skill>/scripts/inspect_exports.py ~/Downloads/snapflow
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

MANIFEST = "manifest.json"
EXPORTS = "exports"
# Suffixes OpenVINO's frontend can read directly, giving I/O shapes and an op count
# without materializing weights.
_READABLE = {".xml", ".onnx"}
# An action chunk output is rank 3: [batch, chunk, action_dim].
_CHUNK_RANK = 3


def discover(root: Path) -> list[tuple[str, Path]]:
    """Find every backend export directory under *root*.

    Returns:
        ``(label, backend_dir)`` pairs, sorted, where label is relative to *root*.
    """
    found = []
    for manifest in sorted(root.rglob(f"{EXPORTS}/*/{MANIFEST}")):
        backend_dir = manifest.parent
        found.append((str(backend_dir.relative_to(root)), backend_dir))
    return found


def check(backend_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """Sanity-check one export against the runtime's own manifest loader.

    Returns:
        ``(manifest_dict, problems)``. The dict is None when the manifest itself
        could not be loaded, in which case the export is unusable.
    """
    problems: list[str] = []
    try:
        from physicalai.inference.manifest import Manifest

        manifest = Manifest.load(backend_dir)
    except Exception as error:
        return None, [f"manifest does not load: {error}"]

    if not manifest.model.artifacts:
        problems.append("manifest declares no 'model.artifacts'")
    for name, rel in manifest.model.artifacts.items():
        if not (backend_dir / rel).is_file():
            problems.append(f"declared artifact '{name}' -> '{rel}' is missing")
    if not manifest.policy.name:
        problems.append("manifest declares no 'policy.name'")

    with (backend_dir / MANIFEST).open(encoding="utf-8") as fh:
        return json.load(fh), problems


def _shape(port: Any) -> list[str]:
    """Render a port's partial shape as plain dimension strings.

    Returns:
        One entry per axis; "?" for a dynamic dimension.
    """
    return [("?" if d.is_dynamic else str(d.get_length())) for d in port.get_partial_shape()]


def introspect(backend_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Read graph shape/size from whichever declared artifact is machine-readable.

    Returns:
        A dict with ``ops``, ``inputs``, ``outputs`` and ``chunk``; empty when no
        declared artifact is in a format that can be introspected.
    """
    artifacts = (manifest.get("model") or {}).get("artifacts") or {}
    for rel in artifacts.values():
        path = backend_dir / rel
        if path.suffix.lower() not in _READABLE or not path.is_file():
            continue
        try:
            import openvino as ov

            model = ov.Core().read_model(str(path))
        except Exception:
            continue
        inputs = [(p.get_any_name(), _shape(p)) for p in model.inputs]
        outputs = [(p.get_any_name(), _shape(p)) for p in model.outputs]
        # The action chunk is the 3-D output [1, chunk, action_dim]; prefer one named
        # "action" and fall back to the first 3-D output.
        chunk = None
        ranked = sorted(outputs, key=lambda o: ("action" not in o[0].lower(), o[0]))
        for _, shape in ranked:
            if len(shape) == _CHUNK_RANK and all(d.isdigit() for d in shape[1:]):
                chunk = (int(shape[1]), int(shape[2]))
                break
        return {"artifact": rel, "ops": len(model.get_ops()), "inputs": inputs, "outputs": outputs, "chunk": chunk}
    return {}


def report(label: str, backend_dir: Path, fps: float, *, deep: bool) -> dict[str, Any] | None:
    """Print one export's report block.

    Returns:
        The manifest dict, or None when the export is unusable.
    """
    manifest, problems = check(backend_dir)
    size = sum(f.stat().st_size for f in backend_dir.rglob("*") if f.is_file())
    print(f"* {label}")
    if manifest is None:
        print(f"    UNUSABLE: {problems[0]}")
        return None

    policy = (manifest.get("policy") or {}).get("name") or "?"
    print(f"    policy   {policy}    on-disk {size / 1e9:.2f} GB")
    pre = [p.get("type") or p.get("class_path") for p in (manifest.get("model") or {}).get("preprocessors", [])]
    if pre:
        print(f"    pipeline {' -> '.join(str(p) for p in pre)}")

    info = introspect(backend_dir, manifest) if deep else {}
    if info:
        print(f"    graph    {info['ops']} ops   (from {info['artifact']})")
        for name, shape in info["inputs"]:
            print(f"      in   {name:<24} {shape}")
        for name, shape in info["outputs"]:
            print(f"      out  {name:<24} {shape}")
        if info["chunk"]:
            chunk, dim = info["chunk"]
            budget = chunk * 0.5 / fps
            print(
                f"    chunk    {chunk} actions x {dim} dims -> ~{budget * 1000:.0f} ms inference budget at {fps:g} Hz",
            )
    elif deep:
        print("    graph    not introspectable (no .xml/.onnx artifact) — compare on size only")

    for problem in problems:
        print(f"    PROBLEM  {problem}")
    return manifest


def main() -> int:
    """Inspect each directory given on the command line.

    Returns:
        Process exit code: 1 if any discovered export is unusable.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--deep",
        action="store_true",
        help="also read I/O shapes and op count from the graph (only when comparing candidates)",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="control loop rate for the budget hint")
    args = parser.parse_args()

    manifests: dict[str, dict[str, Any]] = {}
    bad = False
    for raw in args.paths:
        root = raw.expanduser()
        if not root.is_dir():
            print(f"Not a directory: {root}", file=sys.stderr)
            return 1
        found = discover(root)
        if not found:
            print(f"No '{EXPORTS}/<backend>/{MANIFEST}' anywhere under {root}", file=sys.stderr)
            return 1
        print(f"{root} — {len(found)} export(s)\n")
        for label, backend_dir in found:
            manifest = report(label, backend_dir, args.fps, deep=args.deep)
            if manifest is None:
                bad = True
            else:
                manifests[f"{root}/{label}"] = manifest
            print()

    if len(manifests) > 1:
        keys = list(manifests)
        first = manifests[keys[0]]
        same = all(manifests[k] == first for k in keys[1:])
        print(f"manifests identical across candidates: {same}")
        if same:
            print("So they are drop-in interchangeable; pick on provenance unless you have a reason to dig.")
        else:
            policies = {(m.get("policy") or {}).get("name") for m in manifests.values()}
            pre = {json.dumps((m.get("model") or {}).get("preprocessors"), sort_keys=True) for m in manifests.values()}
            print(f"  policies: {sorted(str(p) for p in policies)}")
            print(f"  preprocessing (incl. normalization stats) identical: {len(pre) == 1}")
        if not args.deep:
            print("Re-run with --deep to compare I/O shapes and graph size.")

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

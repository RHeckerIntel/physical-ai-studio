#!/usr/bin/env python
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""List or fetch policy exports held in a Hugging Face repo.

``list`` enumerates every ``.../exports/<backend>/manifest.json`` in the repo and reports
what each one is, downloading nothing but the manifests themselves (a few KB). ``fetch``
then pulls exactly one candidate.

Run from ``application/backend/`` with that project's interpreter, so Studio's stored
Hugging Face token is picked up for private repos::

    .venv/bin/python <skill>/scripts/hf_exports.py list Daankrol/snapflow-best-ckpt
    .venv/bin/python <skill>/scripts/hf_exports.py fetch Daankrol/snapflow-best-ckpt \
        --candidate snapflow-epoch007 --dest ~/Downloads/snapflow
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

MANIFEST = "manifest.json"
EXPORTS = "exports"
# ".../exports/<backend>/manifest.json" -- the shortest path that can be a candidate.
_MIN_PARTS = 3


def _token() -> str | None:
    """Return Studio's stored Hugging Face token, or None when unset.

    Falls back to the ambient huggingface_hub credentials when Studio's settings
    cannot be imported (e.g. run outside application/backend).

    Returns:
        The token string, or None.
    """
    try:
        from settings import get_settings

        secret = get_settings().huggingface.hf_token
    except Exception:
        return None
    return secret.get_secret_value() if secret else None


def _candidates(files: list[str]) -> dict[str, list[str]]:
    """Group manifest paths by candidate prefix.

    A candidate is whatever precedes ``exports/<backend>/manifest.json``; the empty
    string means the repo root is itself a single exported model.

    Returns:
        Mapping of candidate prefix to its backend names, both sorted.
    """
    found: dict[str, list[str]] = {}
    for path in files:
        parts = path.split("/")
        if len(parts) < _MIN_PARTS or parts[-1] != MANIFEST or parts[-3] != EXPORTS:
            continue
        prefix = "/".join(parts[:-3])
        found.setdefault(prefix, []).append(parts[-2])
    return {k: sorted(v) for k, v in sorted(found.items())}


def _describe(manifest: dict[str, Any], sizes: dict[str, int], prefix: str, backend: str) -> str:
    """Render one candidate/backend as a short report block.

    Returns:
        The formatted block.
    """
    policy = (manifest.get("policy") or {}).get("name") or "?"
    artifacts = (manifest.get("model") or {}).get("artifacts") or {}
    base = f"{prefix + '/' if prefix else ''}{EXPORTS}/{backend}"
    declared = {rel: name for name, rel in artifacts.items()}
    lines = [f"  backend {backend:<12} policy={policy}"]

    # Every file under the backend directory, not just the manifest-declared entry:
    # weights and tokenizers ride alongside as sidecars (an OpenVINO .xml is nothing
    # without its .bin) and they are the real download cost.
    total = 0
    for path in sorted(p for p in sizes if p.startswith(f"{base}/")):
        rel = path[len(base) + 1 :]
        size = sizes[path]
        total += size
        tag = declared.get(rel, "")
        lines.append(f"      {tag:<12} {rel:<28} {size / 1e6:>10.1f} MB")

    missing = [r for r in declared if f"{base}/{r}" not in sizes]
    if missing:
        lines.append(f"      !! declared but absent from the repo: {', '.join(missing)}")
    if total:
        lines.append(f"      {'download':<12} {'':<28} {total / 1e6:>10.1f} MB")
    return "\n".join(lines)


def cmd_list(args: argparse.Namespace) -> int:
    """Print every export found in the repo.

    Returns:
        Process exit code.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=_token())
    files = api.list_repo_files(args.repo, revision=args.revision)
    candidates = _candidates(files)
    if not candidates:
        print(f"No '<candidate>/{EXPORTS}/<backend>/{MANIFEST}' found in {args.repo}.", file=sys.stderr)
        print("Files present:", file=sys.stderr)
        for f in files[:40]:
            print(f"  {f}", file=sys.stderr)
        return 1

    sizes: dict[str, int] = {}
    try:
        for entry in api.list_repo_tree(args.repo, recursive=True, revision=args.revision):
            size = getattr(entry, "size", None)
            if size is not None:
                sizes[entry.path] = size
    except Exception:  # sizes are a nicety, not a requirement
        pass

    print(f"{args.repo} — {len(candidates)} candidate(s)\n")
    for prefix, backends in candidates.items():
        print(f"* {prefix or '<repo root>'}")
        for backend in backends:
            remote = f"{prefix + '/' if prefix else ''}{EXPORTS}/{backend}/{MANIFEST}"
            local = hf_hub_download(args.repo, remote, revision=args.revision, token=_token())
            with Path(local).open(encoding="utf-8") as fh:
                print(_describe(json.load(fh), sizes, prefix, backend))
        print()

    print("Fetch one with:")
    first = next(iter(candidates))
    print(f"  {sys.argv[0]} fetch {args.repo} --candidate {first or '.'} --dest <dir>")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Download exactly one candidate into --dest.

    Returns:
        Process exit code.
    """
    from huggingface_hub import snapshot_download

    candidate = "" if args.candidate in {".", ""} else args.candidate.strip("/")
    patterns = [f"{candidate}/**"] if candidate else ["**"]
    dest = Path(args.dest).expanduser()
    path = snapshot_download(
        args.repo,
        revision=args.revision,
        token=_token(),
        local_dir=str(dest),
        allow_patterns=patterns,
    )
    imported = Path(path) / candidate if candidate else Path(path)
    print(f"Downloaded to {imported}")
    print("Inspect it with: inspect_exports.py", imported)
    return 0


def main() -> int:
    """Parse arguments and dispatch.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="enumerate exports without downloading weights")
    p_list.add_argument("repo")
    p_list.add_argument("--revision", default=None)
    p_list.set_defaults(func=cmd_list)

    p_fetch = sub.add_parser("fetch", help="download one candidate")
    p_fetch.add_argument("repo")
    p_fetch.add_argument("--candidate", required=True, help="prefix from `list`, or '.' for the repo root")
    p_fetch.add_argument("--dest", required=True)
    p_fetch.add_argument("--revision", default=None)
    p_fetch.set_defaults(func=cmd_fetch)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

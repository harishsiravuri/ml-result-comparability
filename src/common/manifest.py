# PROVENANCE: copied verbatim 2026-06-18 from paper2_contribqa/src/common/manifest.py
#   (Paper 2 ContribQA infrastructure).
# The source repository is READ-ONLY; this is the working copy for paper3.2
#   (comparekg, Chapter 3 cross-paper result-cell disagreement census).
# Do not edit the source; edit this copy if behavior must change here.
"""sha256 manifests for everything under data/raw/ (raw files are gitignored;
manifests are committed so the exact bytes used are pinned)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_manifest(manifest_path: Path, files: list[Path], source: str, notes: str = "") -> dict:
    """Write a manifest JSON listing sha256 + size + source URL for each file."""
    entries = []
    for f in sorted(files):
        entries.append(
            {
                "path": str(f),
                "sha256": sha256_file(f),
                "size_bytes": f.stat().st_size,
            }
        )
    doc = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "notes": notes,
        "files": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(doc, indent=2))
    return doc


def verify_manifest(manifest_path: Path) -> list[str]:
    """Return list of mismatch descriptions (empty = all good)."""
    doc = json.loads(manifest_path.read_text())
    problems = []
    for e in doc["files"]:
        p = Path(e["path"])
        if not p.exists():
            problems.append(f"missing: {p}")
        elif sha256_file(p) != e["sha256"]:
            problems.append(f"sha256 mismatch: {p}")
    return problems

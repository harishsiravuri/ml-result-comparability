"""Phase 1: build the candidate-inconsistency set (deterministic, ~$0).

Emits data/census/candidates.jsonl (one CandidatePair per line, full provenance) plus
data/census/candidates_summary.json and a frozen dataset-level dev/test split
(data/census/split.json). Idempotent: re-running reproduces byte-identical output.

Usage:
  python scripts/build_candidates.py [--resume]
  --resume : skip rebuild if candidates.jsonl exists and the input tuple sha matches.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.manifest import sha256_file  # noqa: E402
from common.paths import CENSUS, EXTRACTIONS, ensure_dirs  # noqa: E402
from census.surface import DEV_FRACTION, SPLIT_SEED, surface_candidates  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    ensure_dirs()

    tuples_path = EXTRACTIONS / "tuples.parquet"
    tuples_sha = sha256_file(tuples_path)
    out = CENSUS / "candidates.jsonl"
    cfg_path = CENSUS / "run_config.json"

    if args.resume and out.exists() and cfg_path.exists():
        prev = json.loads(cfg_path.read_text())
        if prev.get("tuples_sha256") == tuples_sha:
            print(f"[resume] up to date ({out}); tuples sha unchanged. Nothing to do.")
            return

    pairs = surface_candidates(tuples_path)

    with open(out, "w") as f:
        for p in pairs:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    # ---- summary (stratified per RULING 1 + RULING 7) ----
    by_type = Counter(p.pair_type for p in pairs)
    by_grade = Counter(p.identity_grade for p in pairs)
    by_split = Counter(p.split for p in pairs)
    by_task = Counter(p.task_family for p in pairs)
    cells = {(p.method_id, p.dataset_id, p.metric_id) for p in pairs}
    cells_by_split = defaultdict(set)
    datasets_by_split = defaultdict(set)
    for p in pairs:
        cells_by_split[p.split].add((p.method_id, p.dataset_id, p.metric_id))
        datasets_by_split[p.split].add(p.dataset_id)
    n_reconciled = sum(1 for p in pairs if p.unit_scale_reconciled)
    with_comember = sum(1 for p in pairs if p.n_protocols_on_dataset_metric > 0)

    summary = {
        "n_candidate_pairs": len(pairs),
        "n_candidate_cells": len(cells),
        "split_seed": SPLIT_SEED, "dev_fraction": DEV_FRACTION, "split_unit": "canonical dataset_id",
        "pairs_by_type": dict(by_type),
        "pairs_by_identity_grade": dict(by_grade),
        "pairs_by_split": dict(by_split),
        "cells_by_split": {k: len(v) for k, v in cells_by_split.items()},
        "datasets_by_split": {k: len(v) for k, v in datasets_by_split.items()},
        "unit_scale_reconciled_pairs": n_reconciled,
        "pairs_with_pwc_comembership": with_comember,
        "top_task_families": dict(sorted(by_task.items(), key=lambda kv: -kv[1])[:20]),
        "tuples_sha256": tuples_sha,
    }
    (CENSUS / "candidates_summary.json").write_text(json.dumps(summary, indent=2))
    cfg_path.write_text(json.dumps({
        "script": "scripts/build_candidates.py", "tuples_sha256": tuples_sha,
        "split_seed": SPLIT_SEED, "dev_fraction": DEV_FRACTION,
        "snapshot": "pwc-archive 2025-07-28",
    }, indent=2))

    # frozen split file (dataset_id -> dev|test), committed so test is frozen
    split_map = {}
    for p in pairs:
        split_map[p.dataset_id] = p.split
    (CENSUS / "split.json").write_text(json.dumps(
        {"seed": SPLIT_SEED, "dev_fraction": DEV_FRACTION, "unit": "dataset_id",
         "assignments": split_map}, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out} ({len(pairs):,} pairs), candidates_summary.json, split.json")


if __name__ == "__main__":
    main()

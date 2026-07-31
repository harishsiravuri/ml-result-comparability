"""Phase 1 diagnostics (deterministic, ~$0) to inform the preregistered bars.

Computes, over the surfaced candidate set:
  - task-family concentration (top-k share, HHI) -> shows the census is NOT solely the
    KGE filtered-vs-raw MRR phenomenon (RULING 7);
  - auto-derivable rule fire rates -> how often the deterministic cross-check can fire
    (split-surface-differs; metric-variant surface-differs), which bounds the
    auto-derivable agreement number and how much cause attribution needs the LLM;
  - value-selection sensitivity (primary vs median vs best-gap): differing-pair counts
    and beyond-coarse-screen counts under each rule (RULING 2 sensitivity);
  - dev/test split integrity (no dataset straddles both splits) and gold-power sketch.

Writes experiments/runs/phase1_diag.json + prints a summary. Reads only structured data.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.paths import CENSUS, EXTRACTIONS, RUNS  # noqa: E402
from census.surface import _norm, reconcile, select_representative  # noqa: E402

COARSE = 0.02
_EPS = 1e-9


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _rel(a, b):
    xa, xb, _ = reconcile(a, b)
    return abs(xa - xb) / max(abs(xa), abs(xb), _EPS)


def main() -> None:
    pairs = [json.loads(l) for l in open(CENSUS / "candidates.jsonl") if l.strip()]
    out = RUNS / "phase1_diag"
    out.mkdir(parents=True, exist_ok=True)

    # ---- task-family concentration ----
    fam = Counter(p["task_family"] for p in pairs)
    total = sum(fam.values())
    top1 = fam.most_common(1)[0]
    top5_share = sum(c for _, c in fam.most_common(5)) / total
    hhi = sum((c / total) ** 2 for c in fam.values())
    # group the famous KGE phenomenon explicitly
    kge_like = {"link prediction", "knowledge base completion", "knowledge graph completion",
                "knowledge graph embedding", "node classification", "semi-supervised node classification"}
    kge_share = sum(c for f, c in fam.items() if f in kge_like) / total

    # ---- auto-derivable rule fire rates ----
    split_differs = 0       # both splits known and differ
    split_known_both = 0
    metric_surface_differs = 0   # same metric_id, different raw metric surface (variant cue)
    for p in pairs:
        ls, rs = _norm(p["left"]["split"]), _norm(p["right"]["split"])
        if ls and rs:
            split_known_both += 1
            if ls != rs:
                split_differs += 1
        if _norm(p["left"]["metric"]) != _norm(p["right"]["metric"]):
            metric_surface_differs += 1
    n = len(pairs)

    # ---- value-selection sensitivity (primary vs median vs best-gap) ----
    df = pd.read_parquet(EXTRACTIONS / "tuples.parquet")
    df = df[df["value"].map(_finite)].copy()
    df["value"] = df["value"].astype(float)
    df["cell"] = list(zip(df["method_id"], df["dataset_id"], df["metric_id"]))
    sel = {"primary": [0, 0], "median": [0, 0], "best_gap": [0, 0]}  # [differing_pairs, beyond_coarse]
    for cell, g in df.groupby("cell", sort=False):
        if g["paper_id"].nunique() < 2:
            continue
        per_paper_rows = {pid: list(pg.itertuples(index=False)) for pid, pg in g.groupby("paper_id")}
        prim = {pid: float(select_representative(rows).value) for pid, rows in per_paper_rows.items()}
        med = {pid: float(pd.Series([r.value for r in rows]).median()) for pid, rows in per_paper_rows.items()}
        valsets = {pid: [r.value for r in rows] for pid, rows in per_paper_rows.items()}
        for pa, pb in combinations(sorted(per_paper_rows), 2):
            # primary
            r = _rel(prim[pa], prim[pb])
            if r > _EPS: sel["primary"][0] += 1
            if r > COARSE: sel["primary"][1] += 1
            # median
            r = _rel(med[pa], med[pb])
            if r > _EPS: sel["median"][0] += 1
            if r > COARSE: sel["median"][1] += 1
            # best-gap: max rel over all value combos
            rb_max = max(_rel(va, vb) for va in valsets[pa] for vb in valsets[pb])
            if rb_max > _EPS: sel["best_gap"][0] += 1
            if rb_max > COARSE: sel["best_gap"][1] += 1

    # ---- split integrity + gold-power sketch ----
    split_map = json.load(open(CENSUS / "split.json"))["assignments"]
    ds_split = defaultdict(set)
    for p in pairs:
        ds_split[p["dataset_id"]].add(p["split"])
    straddlers = [d for d, s in ds_split.items() if len(s) > 1]
    by_split = Counter(p["split"] for p in pairs)
    # projected stratified gold sample of 180: proportional test share
    gold_n = 180
    test_share = by_split["test"] / n
    proj_test_gold = round(gold_n * test_share)

    diag = {
        "n_pairs": n, "n_cells": len({(p["method_id"], p["dataset_id"], p["metric_id"]) for p in pairs}),
        "task_family": {
            "n_families": len(fam), "top1": {"family": top1[0], "share": round(top1[1] / total, 3)},
            "top5_share": round(top5_share, 3), "HHI": round(hhi, 4),
            "kge_like_share": round(kge_share, 3),
            "interpretation": "HHI<<0.15 and top1<25% => broad spread; report kge_like_share plainly",
        },
        "auto_derivable_rules": {
            "split_known_both": split_known_both,
            "split_differs": split_differs,
            "split_differs_rate_of_known": round(split_differs / max(split_known_both, 1), 3),
            "split_differs_rate_of_all": round(split_differs / n, 3),
            "metric_surface_differs": metric_surface_differs,
            "metric_surface_differs_rate": round(metric_surface_differs / n, 3),
        },
        "value_selection_sensitivity": {
            k: {"differing_pairs": v[0], "beyond_coarse_rel_gt_0.02": v[1]} for k, v in sel.items()
        },
        "split_integrity": {
            "datasets_straddling_dev_and_test": len(straddlers),
            "pairs_by_split": dict(by_split),
        },
        "gold_power_sketch": {
            "planned_gold_n": gold_n, "test_pair_share": round(test_share, 3),
            "projected_test_gold_pairs": proj_test_gold,
            "note": "F1 on ~%d test-gold pairs is workable but tight; report Wilson CIs." % proj_test_gold,
        },
    }
    (out / "phase1_diag.json").write_text(json.dumps(diag, indent=2))
    print(json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()

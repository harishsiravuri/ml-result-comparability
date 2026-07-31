"""Gate 0 (KILL-GATE): quantify the candidate-inconsistency population.

Phase 0 feasibility analysis. Uses ONLY the existing structured layer
(data/extractions/tuples.parquet + canon tables + PwC leaderboards). NO LLM calls,
deterministic, ~$0. Question: is there a materially non-trivial population of candidate
beyond-noise cross-paper result-cell disagreements to build a census on?

DEFINITIONS
  cell identity      : canonical (method_id, dataset_id, metric_id).
  candidate cell     : an identity reported by >= 2 DISTINCT papers with differing value.
  paper-pair         : an unordered pair of DISTINCT papers that both report a cell.
                       (Primary unit: cleaner than tuple-pairs, which a single paper can
                        inflate by listing a cell many times.)
  coarse noise screen: a paper-pair "disagrees beyond coarse noise" if its reconciled
                       relative gap exceeds a threshold. The REAL, preregistered noise
                       model is Phase 2; this is only a feasibility screen.

CLEANING (deterministic, reported, conservative — we under- rather than over-count)
  - drop tuples with null/non-finite value.
  - unit-scale reconciliation: when two values differ by a ~100x factor (50x..200x),
    treat it as a percent-vs-fraction unit artifact and bring to a common scale before
    measuring the gap (so 94.7 vs 0.947 is NOT counted as a disagreement). Count how
    often this fires.
  - report an "extraction-quality-clean" view that drops paper-pairs in which either
    tuple was marked UNSUPPORTED by the Paper 2 critic (those are extraction errors,
    not cross-paper disagreement).

OUTPUT
  experiments/runs/gate0/gate0_summary.json (+ run_config.json), and a printed summary.
  The gate verdict and the pre-committed floor are recorded in checkpoints/phase0_report.md.
"""

from __future__ import annotations

import gzip
import json
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.paths import EXTRACTIONS, GOLD, INDEX, RUNS  # noqa: E402
from common.metric_direction import metric_direction  # noqa: E402

# --- pre-committed Gate-0 floor (see REPORT.md / PREREGISTRATION will re-state) ---
GATE0_THRESHOLD = 0.02          # 2% reconciled relative gap = coarse "beyond seed noise"
GATE0_FLOOR_PAIRS = 300         # >= this many disagreeing cross-paper paper-pairs to PASS
REL_THRESHOLDS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]
ABS_PP_THRESHOLDS = [0.0, 0.5, 1.0, 2.0, 5.0]   # for the both-percent subset


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def reconcile(v1: float, v2: float) -> tuple[float, float, bool]:
    """Bring two values to a common scale when they look like a ~100x unit artifact."""
    a, b = abs(v1), abs(v2)
    hi, lo = max(a, b), min(a, b)
    if lo > 0 and 50.0 <= hi / lo <= 200.0:
        # the larger is on a %-scale, the smaller a fraction of the same quantity
        if a < b:
            return v1 * 100.0, v2, True
        return v1, v2 * 100.0, True
    return v1, v2, False


def rel_gap(v1: float, v2: float) -> float:
    denom = max(abs(v1), abs(v2), 1e-9)
    return abs(v1 - v2) / denom


def identity_grade(mid: str, did: str, met: str) -> str:
    nonhash = [not str(x).startswith("hash:") for x in (mid, did, met)]
    if all(nonhash):
        return "all_pwc"
    if any(nonhash):
        return "partial_pwc"
    return "hash_only"


def main() -> None:
    out_dir = RUNS / "gate0"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(EXTRACTIONS / "tuples.parquet")
    n_tuples_total = len(df)
    df = df[df["value"].map(_finite)].copy()
    df["value"] = df["value"].astype(float)
    n_with_value = len(df)
    n_papers = df["paper_id"].nunique()

    # critic-clean flag: a tuple is an extraction error if the critic refuted it.
    df["extr_ok"] = df["critic_verdict"].fillna("").map(lambda v: v != "UNSUPPORTED")

    # group by canonical identity
    df["cell"] = list(zip(df["method_id"], df["dataset_id"], df["metric_id"]))
    n_cells_total = df["cell"].nunique()

    # per-cell, collapse to per-paper value lists (so one paper can't inflate pairs)
    # keep a representative quote/grade per (cell, paper) for downstream provenance sanity.
    cell_groups: dict[tuple, dict] = {}
    for cell, g in df.groupby("cell", sort=False):
        papers = g["paper_id"].nunique()
        if papers < 2:
            continue
        mid, did, met = cell
        per_paper: dict[str, list] = defaultdict(list)
        for r in g.itertuples(index=False):
            per_paper[r.paper_id].append((r.value, bool(r.unit == "%"), bool(r.extr_ok), bool(r.is_own_result)))
        cell_groups[cell] = {
            "n_papers": papers,
            "metric_name": g["metric"].iloc[0],
            "dataset_name": g["dataset"].iloc[0],
            "method_name": g["method"].iloc[0],
            "grade": identity_grade(mid, did, met),
            "per_paper": dict(per_paper),
        }

    n_cells_multi_paper = len(cell_groups)

    # ---- screens ----
    # For each multi-paper cell, examine cross-paper paper-pairs at the closest (min-gap)
    # reconciled comparison between the two papers' value sets.
    rel_curve = {f"{t}": {"cells": 0, "paper_pairs": 0, "paper_pairs_own": 0} for t in REL_THRESHOLDS}
    abs_curve = {f"{t}": {"cells": 0, "paper_pairs": 0} for t in ABS_PP_THRESHOLDS}
    grade_at_floor: dict[str, int] = defaultdict(int)         # grade -> #cells disagreeing at floor
    n_scale_reconciled_pairs = 0
    extr_clean_floor_pairs = 0
    group_size_hist: dict[int, int] = defaultdict(int)
    examples: list[dict] = []
    disagreeing_cell_datasets: set[str] = set()               # datasets of cells crossing the floor

    for cell, info in cell_groups.items():
        group_size_hist[min(info["n_papers"], 50)] += 1
        papers = list(info["per_paper"].items())
        # per-cell tallies of whether ANY cross-paper pair crosses each threshold
        cell_hits_rel = {t: False for t in REL_THRESHOLDS}
        cell_hits_abs = {t: False for t in ABS_PP_THRESHOLDS}
        cell_max_rel = 0.0
        cell_floor_pp = 0
        for (pa, va_list), (pb, vb_list) in combinations(papers, 2):
            # best (smallest) reconciled gap between the two papers (most conservative)
            best_rel = None
            best_abs = None
            best_recpair = None
            both_own = False
            both_extr_ok = False
            for (va, va_pct, va_ok, va_own) in va_list:
                for (vb, vb_pct, vb_ok, vb_own) in vb_list:
                    ra, rb, adj = reconcile(va, vb)
                    rg = rel_gap(ra, rb)
                    if best_rel is None or rg < best_rel:
                        best_rel = rg
                        best_recpair = (ra, rb, adj, va, vb)
                        both_own = va_own and vb_own
                        both_extr_ok = va_ok and vb_ok
                        if va_pct and vb_pct:
                            best_abs = abs(va - vb)
                        else:
                            best_abs = None
            ra, rb, adj, va, vb = best_recpair
            if adj:
                n_scale_reconciled_pairs += 1
            for t in REL_THRESHOLDS:
                if best_rel > t:
                    rel_curve[f"{t}"]["paper_pairs"] += 1
                    cell_hits_rel[t] = True
                    if both_own:
                        rel_curve[f"{t}"]["paper_pairs_own"] += 1
            if best_abs is not None:
                for t in ABS_PP_THRESHOLDS:
                    if best_abs > t:
                        abs_curve[f"{t}"]["paper_pairs"] += 1
                        cell_hits_abs[t] = True
            if best_rel > GATE0_THRESHOLD:
                cell_floor_pp += 1
                if both_extr_ok:
                    extr_clean_floor_pairs += 1
            cell_max_rel = max(cell_max_rel, best_rel)
        for t in REL_THRESHOLDS:
            if cell_hits_rel[t]:
                rel_curve[f"{t}"]["cells"] += 1
        for t in ABS_PP_THRESHOLDS:
            if cell_hits_abs[t]:
                abs_curve[f"{t}"]["cells"] += 1
        if cell_floor_pp > 0:
            grade_at_floor[info["grade"]] += 1
            disagreeing_cell_datasets.add(str(info["dataset_name"]).strip().lower())
            if len(examples) < 25:
                examples.append({
                    "method": info["method_name"], "dataset": info["dataset_name"],
                    "metric": info["metric_name"], "n_papers": info["n_papers"],
                    "max_rel_gap": round(cell_max_rel, 4), "grade": info["grade"],
                    "metric_direction": metric_direction(str(info["metric_name"])),
                })

    floor_pairs = rel_curve[f"{GATE0_THRESHOLD}"]["paper_pairs"]
    floor_cells = rel_curve[f"{GATE0_THRESHOLD}"]["cells"]
    passed = floor_pairs >= GATE0_FLOOR_PAIRS

    # ---- leaderboard-audit ceiling preview: how many disagreeing cells map to a PwC
    # leaderboard by (dataset name)? (rough co-membership upper bound; refined in Phase 5)
    lb_datasets = set()
    with gzip.open(GOLD / "leaderboards_all.jsonl.gz", "rt") as f:
        for line in f:
            d = json.loads(line)
            if d.get("dataset"):
                lb_datasets.add(str(d["dataset"]).strip().lower())
    disagreeing_datasets_on_a_pwc_lb = sorted(disagreeing_cell_datasets & lb_datasets)

    summary = {
        "snapshot": "pwc-archive frozen 2025-07-28 (service shutdown 2025-07-24)",
        "inputs": {
            "n_tuples_total": int(n_tuples_total),
            "n_tuples_with_finite_value": int(n_with_value),
            "n_papers_with_tuples": int(n_papers),
            "n_cells_total": int(n_cells_total),
            "n_cells_multi_paper": int(n_cells_multi_paper),
        },
        "gate0": {
            "threshold_rel_gap": GATE0_THRESHOLD,
            "floor_pairs_required": GATE0_FLOOR_PAIRS,
            "disagreeing_paper_pairs_at_floor": int(floor_pairs),
            "disagreeing_cells_at_floor": int(floor_cells),
            "extraction_clean_paper_pairs_at_floor": int(extr_clean_floor_pairs),
            "PASSED": bool(passed),
        },
        "rel_gap_curve": rel_curve,
        "abs_pp_curve_both_percent": abs_curve,
        "unit_scale_reconciled_paper_pairs": int(n_scale_reconciled_pairs),
        "identity_grade_at_floor_cells": dict(grade_at_floor),
        "papers_per_cell_hist_capped50": {str(k): v for k, v in sorted(group_size_hist.items())},
        "example_disagreeing_cells": examples,
        "leaderboard_audit_preview": {
            "n_pwc_leaderboard_datasets": len(lb_datasets),
            "n_disagreeing_cell_datasets": len(disagreeing_cell_datasets),
            "n_disagreeing_datasets_on_a_pwc_leaderboard": len(disagreeing_datasets_on_a_pwc_lb),
        },
    }
    (out_dir / "gate0_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "run_config.json").write_text(json.dumps({
        "script": "scripts/gate0_candidate_population.py",
        "seed": "n/a (deterministic)",
        "snapshot": "pwc-archive 2025-07-28",
        "tuples_sha_note": "see data/SNAPSHOT.md",
        "threshold": GATE0_THRESHOLD, "floor": GATE0_FLOOR_PAIRS,
    }, indent=2))

    # ---- printed summary ----
    print("=" * 72)
    print("GATE 0 — candidate-inconsistency population (feasibility, $0, deterministic)")
    print("=" * 72)
    print(f"tuples: {n_with_value:,}/{n_tuples_total:,} with finite value | papers: {n_papers:,}")
    print(f"cells (canonical identities): {n_cells_total:,} total | {n_cells_multi_paper:,} multi-paper")
    print(f"unit-scale (~100x) reconciled paper-pairs: {n_scale_reconciled_pairs:,}")
    print()
    print("Relative-gap screen (reconciled; cross-paper paper-pairs):")
    print(f"  {'thr':>6} {'cells':>10} {'paper_pairs':>14} {'pairs(both own)':>18}")
    for t in REL_THRESHOLDS:
        r = rel_curve[f"{t}"]
        print(f"  {t:>6} {r['cells']:>10,} {r['paper_pairs']:>14,} {r['paper_pairs_own']:>18,}")
    print()
    print("Absolute points screen (both sides unit=%):")
    print(f"  {'thr_pp':>6} {'cells':>10} {'paper_pairs':>14}")
    for t in ABS_PP_THRESHOLDS:
        a = abs_curve[f"{t}"]
        print(f"  {t:>6} {a['cells']:>10,} {a['paper_pairs']:>14,}")
    print()
    print(f"identity grade of disagreeing cells @floor(rel>{GATE0_THRESHOLD}): {dict(grade_at_floor)}")
    print()
    verdict = "PASS" if passed else "FAIL — STOP and write FAILURE_ANALYSIS.md"
    print(f"GATE 0 floor: >= {GATE0_FLOOR_PAIRS} disagreeing paper-pairs at rel>{GATE0_THRESHOLD}")
    print(f"  observed: {floor_pairs:,} paper-pairs across {floor_cells:,} cells  ->  {verdict}")
    print(f"  (extraction-clean: {extr_clean_floor_pairs:,} paper-pairs)")
    print("=" * 72)
    print(f"wrote {out_dir/'gate0_summary.json'}")


if __name__ == "__main__":
    main()

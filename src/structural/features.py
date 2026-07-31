"""Structural features for the cross-paper comparison-validity detector.

These are the signals a PER-PAIR model cannot see: they are computed from the WHOLE corpus
and the Papers-with-Code structure. Target = validity of a cross-paper COMPARISON (is this a
real, valid disagreement versus within-noise / an extraction-or-identity artifact /
incomparable), NOT single-fact correctness. No Papers-with-Code curated value enters these
features (they use the papers' own reported values and the leaderboard co-membership metadata
only).

Feature groups:
  value-distribution  : for the cell's full corpus value set, where each side sits (robust
                        z-scores), the spread, and whether a side is an extreme outlier.
  identity-ambiguity  : generic method names ("Ours"); a canonical method_id spanning many
                        cells or carrying divergent values (a likely identity-merge artifact).
  co-membership       : how many distinct Papers-with-Code leaderboards (protocols) the
                        dataset+metric hosts (an incomparability prior).
  constraints (fixkg) : metric-bound / magnitude-outlier / intra-paper / multi-report /
                        ranking-inversion / claim-coherence violations touching either side.
  extraction-quality  : per-side critic verdict, quote verification, self-consistency.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from structural.constraint_engine import detect, per_metric_ranges

_GENERIC = {"ours", "our", "our method", "our model", "ours (ours)", "baseline", "model",
            "proposed", "method", "the model", "base", "full", "full model", "single-stage",
            "multi-stage", "ours*", "ours+", "backbone", "network"}


def _generic_name(m: str) -> bool:
    n = re.sub(r"\s+", " ", (m or "").strip().lower())
    if n in _GENERIC or n.startswith("our") or "(ours" in n:
        return True
    core = re.sub(r"[^a-z0-9]", "", n)
    return len(core) <= 2


def build_corpus_index(df: pd.DataFrame) -> dict:
    """Precompute corpus-wide structure once (value stats, id stats, constraint hits)."""
    df = df[df["value"].map(lambda v: pd.notna(v) and math.isfinite(float(v)))].reset_index(drop=True)
    df["value"] = df["value"].astype(float)

    # per-cell value distribution
    cell_stats = {}
    for cell, g in df.groupby(["method_id", "dataset_id", "metric_id"], sort=False):
        v = g["value"].to_numpy(float)
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        cell_stats[cell] = {"n": len(v), "median": med, "mad": mad,
                            "mean": float(np.mean(v)), "std": float(np.std(v)),
                            "min": float(v.min()), "max": float(v.max())}
    # per-method_id spread across the corpus (identity ambiguity)
    method_ncells = defaultdict(set)
    for mid, did, mtid in zip(df["method_id"], df["dataset_id"], df["metric_id"]):
        method_ncells[mid].add((did, mtid))
    method_ncells = {k: len(v) for k, v in method_ncells.items()}

    # constraint engine over the full corpus -> per-tuple violation-type flags
    pmr = per_metric_ranges(df)
    viols = detect(df, pmr)
    tup_viol = defaultdict(set)  # (paper_id,method_id,dataset_id,metric_id,round(value,6)) -> {types}
    for vv in viols:
        for i in vv["members"]:
            r = df.iloc[i]
            key = (r["paper_id"], r["method_id"], r["dataset_id"], r["metric_id"], round(float(r["value"]), 6))
            tup_viol[key].add(vv["type"])
    return {"cell_stats": cell_stats, "method_ncells": method_ncells, "tup_viol": tup_viol}


def _z(v, med, mad):
    if mad <= 1e-9:
        return 0.0
    return (v - med) / (mad * 1.4826)


def _viol_flags(side, idx):
    key = (side["paper_id"], side["method_id"], side["dataset_id"], side["metric_id"],
           round(float(side["value"]), 6))
    return idx["tup_viol"].get(key, set())


FEATURE_NAMES = [
    "cell_n_reports", "cell_log_n", "cell_rel_spread", "max_abs_z", "min_abs_z",
    "either_extreme_outlier", "pair_gap_over_mad", "cell_cv",
    "either_generic_name", "both_generic_name", "method_max_ncells", "method_id_hashed",
    "n_protocols", "on_pwc_leaderboard",
    "vio_metric_bound", "vio_magnitude_outlier", "vio_multi_report", "vio_intra_paper",
    "vio_ranking_inversion", "vio_claim_coherence", "n_vio_types",
    "either_unsupported", "either_low_selfcons", "either_unverified_quote",
    "identity_grade_hash", "unit_scale_reconciled", "rel_gap",
]


def pair_features(pair: dict, idx: dict) -> dict:
    L, R = pair["left"], pair["right"]
    cell = (pair["method_id"], pair["dataset_id"], pair["metric_id"])
    cs = idx["cell_stats"].get(cell, {"n": 2, "median": (L["value"] + R["value"]) / 2,
                                      "mad": abs(L["value"] - R["value"]) / 2 or 1.0,
                                      "mean": (L["value"] + R["value"]) / 2, "std": 0.0,
                                      "min": min(L["value"], R["value"]), "max": max(L["value"], R["value"])})
    lz, rz = _z(L["value"], cs["median"], cs["mad"]), _z(R["value"], cs["median"], cs["mad"])
    scale = max(abs(cs["median"]), 1e-9)
    cv = (cs["std"] / abs(cs["mean"])) if abs(cs["mean"]) > 1e-9 else 0.0
    lf, rf = _viol_flags(L, idx), _viol_flags(R, idx)
    allf = lf | rf
    mid = pair["method_id"]
    return {
        "cell_n_reports": cs["n"], "cell_log_n": math.log1p(cs["n"]),
        "cell_rel_spread": (cs["max"] - cs["min"]) / scale,
        "max_abs_z": max(abs(lz), abs(rz)), "min_abs_z": min(abs(lz), abs(rz)),
        "either_extreme_outlier": float(max(abs(lz), abs(rz)) > 3.5),
        "pair_gap_over_mad": abs(L["value"] - R["value"]) / (cs["mad"] * 1.4826 + 1e-9),
        "cell_cv": cv,
        "either_generic_name": float(_generic_name(L["method"]) or _generic_name(R["method"])),
        "both_generic_name": float(_generic_name(L["method"]) and _generic_name(R["method"])),
        "method_max_ncells": idx["method_ncells"].get(mid, 1),
        "method_id_hashed": float(str(mid).startswith("hash:")),
        "n_protocols": pair.get("n_protocols_on_dataset_metric", 0),
        "on_pwc_leaderboard": float(pair.get("n_protocols_on_dataset_metric", 0) > 0),
        "vio_metric_bound": float("metric_bound" in allf),
        "vio_magnitude_outlier": float("magnitude_outlier" in allf),
        "vio_multi_report": float("multi_report" in allf),
        "vio_intra_paper": float("intra_paper" in allf),
        "vio_ranking_inversion": float("ranking_inversion" in allf),
        "vio_claim_coherence": float("claim_coherence" in allf),
        "n_vio_types": float(len(allf)),
        "either_unsupported": float(L.get("critic_verdict") == "UNSUPPORTED" or R.get("critic_verdict") == "UNSUPPORTED"),
        "either_low_selfcons": float((L.get("self_consistency") or 1.0) < 0.5 or (R.get("self_consistency") or 1.0) < 0.5),
        "either_unverified_quote": float((not L.get("quote_verified")) or (not R.get("quote_verified"))),
        "identity_grade_hash": float(pair.get("identity_grade") == "hash_only"),
        "unit_scale_reconciled": float(pair.get("unit_scale_reconciled", False)),
        "rel_gap": float(pair.get("rel_gap", 0.0)),
    }


def features_vector(pair: dict, idx: dict) -> list[float]:
    f = pair_features(pair, idx)
    return [float(f[k]) for k in FEATURE_NAMES]

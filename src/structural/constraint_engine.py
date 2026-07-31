# PROVENANCE: copied 2026-06-18 from paper2.3/src/fixkg/constraints/engine.py (fixkg symbolic constraint engine), READ-ONLY source (Chapter 2.3 fixkg).
# Reused here as a STRUCTURAL FEATURE source for the Chapter 3 comparison-validity
# detector. Target = validity of a cross-paper COMPARISON, not single-fact correctness.
"""Symbolic constraint engine. Each violation identifies the SET of facts in conflict and,
where a constraint-consistent value exists, a proposed repair. Soft predicates over fact
sets; deterministic. Constraints never read a held-out real-error label.

A Violation is a dict: {type, members: list[int] (positional row indices), repair: float|None,
weight: float}. weight scales the constraint's strength (e.g., multi-report disagreement
magnitude). The engine operates on a positionally-indexed DataFrame with columns:
paper_id, method_id, dataset_id, metric_id, method, dataset, metric, value, is_own_result,
claim_strength.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .semantics import (bounded_violation, error_violation, metric_direction,
                                metric_type, value_close)


def detect(df: pd.DataFrame, per_metric_range: dict | None = None) -> list[dict]:
    """Return all constraint violations over df (df must be 0..n-1 indexed)."""
    df = df.reset_index(drop=True)
    n = len(df)
    viols: list[dict] = []
    val = df["value"].to_numpy(float)

    # (i) metric bounds + (v) magnitude/precision -- singleton violations
    for i in range(n):
        m = df.at[i, "metric"]
        if bounded_violation(m, val[i]):
            rep = val[i] / 100.0 if val[i] > 100 else None
            viols.append({"type": "metric_bound", "members": [i], "repair": rep, "weight": 1.0})
        elif error_violation(m, val[i]):
            viols.append({"type": "metric_bound", "members": [i], "repair": abs(val[i]), "weight": 1.0})
    if per_metric_range:
        for i in range(n):
            r = per_metric_range.get(df.at[i, "metric_id"])
            if r and (val[i] < r["lo"] or val[i] > r["hi"]):
                viols.append({"type": "magnitude_outlier", "members": [i],
                              "repair": float(r["median"]), "weight": 0.5})

    # (iii) intra-paper incoherence: conflicting duplicate cells in one paper
    for _, grp in df.groupby(["paper_id", "method_id", "dataset_id", "metric_id"]):
        if len(grp) < 2:
            continue
        idx = list(grp.index)
        vals = grp["value"].to_numpy(float)
        if not all(value_close(vals[0], v) for v in vals):
            viols.append({"type": "intra_paper", "members": idx,
                          "repair": float(np.median(vals)), "weight": 1.0})

    # claim-vs-number incoherence: own-result improves/sota but not best for its paper-cell
    for _, grp in df.groupby(["paper_id", "dataset_id", "metric_id"]):
        own = grp[grp["is_own_result"] == True]  # noqa: E712
        if own.empty:
            continue
        direction = metric_direction(grp["metric"].iloc[0])
        if direction == "unknown":
            continue
        best = grp["value"].max() if direction == "higher" else grp["value"].min()
        for i, row in own.iterrows():
            if str(row.get("claim_strength")) in ("improves", "sota_claim"):
                is_best = (row["value"] >= best - 1e-9) if direction == "higher" else (row["value"] <= best + 1e-9)
                if not is_best:
                    viols.append({"type": "claim_coherence", "members": [int(i)], "repair": None, "weight": 0.5})

    # (iv) multi-report disagreement across papers
    for _, grp in df.groupby(["method_id", "dataset_id", "metric_id"]):
        if grp["paper_id"].nunique() < 2:
            continue
        idx = list(grp.index)
        vals = grp["value"].to_numpy(float)
        med = np.median(np.abs(vals)) or 1.0
        if (vals.max() - vals.min()) / med > 0.01 and not all(value_close(vals[0], v) for v in vals):
            spread = (vals.max() - vals.min()) / med
            viols.append({"type": "multi_report", "members": idx, "repair": float(np.median(vals)),
                          "weight": float(min(2.0, 0.5 + spread))})

    # (ii) ranking/transitivity: tightened -- flag an inverted pair only when the inversion
    # is driven by a method's OWN cross-paper spread (not genuine closeness of two methods).
    for _, grp in df.groupby(["dataset_id", "metric_id"]):
        if grp["method_id"].nunique() < 3 or grp["paper_id"].nunique() < 2:
            continue
        direction = metric_direction(grp["metric"].iloc[0])
        if direction == "unknown":
            continue
        by_m = defaultdict(list)
        for i, mi, v in zip(grp.index, grp["method_id"], grp["value"]):
            by_m[mi].append((int(i), float(v)))
        # only methods with internal spread can cause a spread-driven inversion
        spread_methods = {mi: vs for mi, vs in by_m.items()
                          if (max(v for _, v in vs) - min(v for _, v in vs)) > 1e-9}
        methods = list(by_m)
        for mi, vs in spread_methods.items():
            mn = min(v for _, v in vs); mx = max(v for _, v in vs)
            for mj in methods:
                if mj == mi:
                    continue
                ovs = by_m[mj]
                for j, ov in ovs:
                    # inversion: mi sometimes above and sometimes below mj's value ov
                    if mn < ov < mx:
                        members = [i for i, _ in vs] + [j]
                        viols.append({"type": "ranking_inversion", "members": sorted(set(members)),
                                      "repair": None, "weight": 0.3})
                        break
    return viols


def per_metric_ranges(df: pd.DataFrame, lo_p=0.5, hi_p=99.5, min_n=20) -> dict:
    out = {}
    for mid, grp in df.groupby("metric_id"):
        if len(grp) < min_n:
            continue
        v = grp["value"].to_numpy(float)
        lo, hi = np.percentile(v, [lo_p, hi_p])
        out[mid] = {"lo": float(lo), "hi": float(hi), "median": float(np.median(v))}
    return out

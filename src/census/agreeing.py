"""Item 1: surface AGREEING same-cell cross-paper pairs (deterministic, NO LLM, $0).

The released census covers DIFFERING-value pairs only. Comparability is a property of
protocols, not of values, so the decision has to hold on agreeing pairs too. This module
surfaces exactly the pairs `census.surface.surface_candidates` drops: cross-paper pairs on a
shared canonical (method, dataset, metric) cell whose values are EQUAL after the same
percent-vs-fraction reconciliation. The two populations therefore partition the same-cell
cross-paper universe with no overlap and no gap.

Every upstream rule is reused unchanged (RULING 2 representative selection, the identity
grade, the frozen dataset-level dev/test split), so an agreeing pair is comparable
like-for-like with a candidate pair.
"""

from __future__ import annotations

import gzip
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from census.surface import (
    _EPS,
    _cell_to_record,
    _finite,
    _norm,
    build_comembership_index,
    identity_grade,
    reconcile,
    select_representative,
    split_for_dataset,
)
from certificates.facets import per_facet, side_protocol, surface_normalizations
from common.metric_direction import metric_direction
from common.paths import EXTRACTIONS, INDEX


def protocol_relation(left: dict, right: dict) -> dict:
    """Observed-facet relation for an agreeing pair, using the item-2 facet layer.

    Goes through per_facet (not the bare `relation`) so the metric-surface alias and
    one-sided-variant corrections apply here too: a pair must not be called cross-protocol
    because one paper wrote "acc" and the other wrote "accuracy".
    """
    lp, rp = side_protocol(left), side_protocol(right)
    rels = {r["facet"]: r["relation"] for r in per_facet(lp, rp)}
    differing = sorted(f for f, r in rels.items() if r == "observed-different")
    missing = sorted(f for f, r in rels.items() if r == "missing")
    if differing:
        klass = "cross_protocol"          # agreement DESPITE an observed protocol difference
    elif missing:
        klass = "protocol_unknown"        # cannot tell: a facet is missing on a side
    else:
        klass = "same_observed_protocol"
    return {"left_protocol": lp, "right_protocol": rp, "facet_relations": rels,
            "differing_facets": differing, "missing_facets": missing,
            "normalizations": surface_normalizations(lp, rp),
            "protocol_class": klass}


def surface_agreeing(tuples_path: Path | None = None) -> list[dict]:
    df = pd.read_parquet(tuples_path or (EXTRACTIONS / "tuples.parquet"))
    df = df[df["value"].map(_finite)].copy()
    df["value"] = df["value"].astype(float)

    canon = json.load(gzip.open(INDEX / "canon_tables.json.gz", "rt"))["entries"]

    def canon_name(cid: str, fallback: str) -> str:
        e = canon.get(str(cid))
        return e["canonical_name"] if e else str(fallback)

    comember = build_comembership_index()
    df["cell"] = list(zip(df["method_id"], df["dataset_id"], df["metric_id"]))

    out: list[dict] = []
    n = 0
    for cell, g in df.groupby("cell", sort=True):
        if g["paper_id"].nunique() < 2:
            continue
        mid, did, met = (str(x) for x in cell)
        reps = {pid: select_representative(list(pg.itertuples(index=False)))
                for pid, pg in g.groupby("paper_id", sort=True)}
        for pa, pb in combinations(sorted(reps), 2):
            ra, rb = reps[pa], reps[pb]
            xa, xb, adjusted = reconcile(float(ra.value), float(rb.value))
            if abs(xa - xb) > _EPS:
                continue                       # differing -> belongs to the candidate set
            n += 1
            left, right = _cell_to_record(ra), _cell_to_record(rb)
            own_a, own_b = bool(ra.is_own_result), bool(rb.is_own_result)
            ptype = ("both_own" if own_a and own_b
                     else "own_vs_cited" if own_a != own_b else "cited_vs_cited")
            lb_ids = comember.get((_norm(g["dataset"].iloc[0]), _norm(g["metric"].iloc[0])), [])
            rec = {
                "pair_id": f"ap{n:06d}", "method_id": mid, "dataset_id": did, "metric_id": met,
                "method_canonical": canon_name(mid, g["method"].iloc[0]),
                "dataset_canonical": canon_name(did, g["dataset"].iloc[0]),
                "metric_canonical": canon_name(met, g["metric"].iloc[0]),
                "metric_direction": metric_direction(str(g["metric"].iloc[0])),
                "left": left.__dict__ if hasattr(left, "__dict__") else left,
                "right": right.__dict__ if hasattr(right, "__dict__") else right,
                "agreeing": True, "value_gap": 0.0,
                "unit_scale_reconciled": bool(adjusted),
                "identity_grade": identity_grade(mid, did, met),
                "pair_type": ptype,
                "task_family": (_norm(ra.task) or _norm(rb.task) or "unknown"),
                "split": split_for_dataset(did),
                "n_protocols_on_dataset_metric": len(lb_ids),
            }
            rec["protocol"] = protocol_relation(rec["left"], rec["right"])
            out.append(rec)
    return out

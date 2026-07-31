"""Phase 1 candidate surfacing (deterministic, NO LLM, ~$0).

Surfaces every cross-paper differing-value pair for a shared canonical
(method, dataset, metric) identity, with full provenance, under the strategic
rulings (relayed 2026-06-18):

  RULING 1 (pair inclusion): include ALL cross-paper differing-value pairs; do NOT
    restrict to both-own. Each pair is typed: both_own | own_vs_cited | cited_vs_cited.
  RULING 2 (within-paper value selection): one representative tuple per (cell, paper)
    by a committed PROVENANCE rule (NOT Gate-0 best-gap, which is adversarial):
      rank by (is_own_result, quote_verified, critic_rank, self_consistency) desc;
      among the top-ranked tier, take the lower-median-by-value tuple (an ACTUAL tuple,
      so its source span is preserved). Sensitivity vs best-gap and vs median is a
      reporting deliverable (src/census/sensitivity in diagnostics), not the primary.
  RULING 7 (honesty): every record carries identity_grade and task_family so prevalence
    can travel with coverage and be broken out across task families.

This module emits CandidatePair records (schema.py). The beyond-noise decision (Phase 2)
and the cause (Phase 3) are added later; Phase 1 does not decide significance.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

from common.metric_direction import metric_direction
from common.paths import EXTRACTIONS, GOLD, INDEX
from census.schema import CandidateCell, CandidatePair

# Dev/test split (frozen). Split at canonical DATASET level so all protocol variants of
# a dataset stay on one side (no protocol leakage across splits). Seed + ratio committed
# in PREREGISTRATION.md.
SPLIT_SEED = 13
DEV_FRACTION = 0.30  # AMENDMENT D (2026-06-18): 30/70 — enlarges test census + test-gold
                     #   power (~140 test-gold), matches the prior chapter; dev ~900 pairs
                     #   remains ample for judge tuning.

_CRITIC_RANK = {"SUPPORTED": 3, "PARTIAL": 2, None: 1, "": 1, "UNSUPPORTED": 0}
_WS = re.compile(r"\s+")
_EPS = 1e-9


def _norm(s) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    out = _WS.sub(" ", str(s).strip().lower())
    return "" if out in ("nan", "none", "null") else out


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def identity_grade(mid: str, did: str, met: str) -> str:
    nonhash = [not str(x).startswith("hash:") for x in (mid, did, met)]
    if all(nonhash):
        return "all_pwc"
    if any(nonhash):
        return "partial_pwc"
    return "hash_only"


def reconcile(v1: float, v2: float) -> tuple[float, float, bool]:
    """Bring two values to a common scale on a ~100x (percent-vs-fraction) artifact."""
    a, b = abs(v1), abs(v2)
    hi, lo = max(a, b), min(a, b)
    if lo > 0 and 50.0 <= hi / lo <= 200.0:
        if a < b:
            return v1 * 100.0, v2, True
        return v1, v2 * 100.0, True
    return v1, v2, False


def _provenance_key(r) -> tuple:
    """Higher is better. r is a namedtuple row from itertuples."""
    cv = _CRITIC_RANK.get(r.critic_verdict if isinstance(r.critic_verdict, str) else "", 1)
    return (
        1 if bool(r.is_own_result) else 0,
        1 if bool(r.quote_verified) else 0,
        cv,
        float(r.self_consistency) if _finite(r.self_consistency) else 0.0,
    )


def select_representative(rows: list) -> object:
    """RULING 2 primary rule. rows: list of itertuples for one (cell, paper)."""
    best_key = max(_provenance_key(r) for r in rows)
    tier = [r for r in rows if _provenance_key(r) == best_key]
    tier_sorted = sorted(tier, key=lambda r: (float(r.value), str(r.evidence_quote)))
    return tier_sorted[(len(tier_sorted) - 1) // 2]  # lower-median actual tuple


def split_for_dataset(dataset_id: str) -> str:
    h = hashlib.sha256(f"{SPLIT_SEED}|{dataset_id}".encode()).hexdigest()
    frac = int(h[:8], 16) / 0xFFFFFFFF
    return "dev" if frac < DEV_FRACTION else "test"


def _cell_to_record(row) -> CandidateCell:
    return CandidateCell(
        paper_id=str(row.paper_id), method_id=str(row.method_id),
        dataset_id=str(row.dataset_id), metric_id=str(row.metric_id),
        method=str(row.method), dataset=str(row.dataset), metric=str(row.metric),
        value=float(row.value), unit=(None if pd.isna(row.unit) else str(row.unit)),
        split=(None if pd.isna(row.split) else str(row.split)),
        task=(None if pd.isna(row.task) else str(row.task)),
        is_own_result=bool(row.is_own_result),
        evidence_quote=str(row.evidence_quote), source_block=str(row.source_block),
        quote_verified=bool(row.quote_verified),
        self_consistency=(None if pd.isna(row.self_consistency) else float(row.self_consistency)),
        critic_verdict=(None if (not isinstance(row.critic_verdict, str)) else row.critic_verdict),
    )


def build_comembership_index() -> dict[tuple[str, str], list[str]]:
    """(norm_dataset, norm_metric) -> [lb_id ...] : the PwC protocol variants for a
    dataset+metric. A coarse incomparability prior (count of distinct protocols);
    refined into the pairwise leaderboard-audit number in Phase 5. Metadata only, never
    a value, never enters a prompt."""
    idx: dict[tuple[str, str], set] = defaultdict(set)
    with gzip.open(GOLD / "leaderboards_all.jsonl.gz", "rt") as f:
        for line in f:
            d = json.loads(line)
            ds = _norm(d.get("dataset"))
            if not ds:
                continue
            metrics = set()
            for row in d.get("rows", []):
                for m in (row.get("metrics") or {}):
                    metrics.add(_norm(m))
            if d.get("metrics"):
                metrics.add(_norm(d.get("metrics")))
            for m in metrics:
                if m:
                    idx[(ds, m)].add(d["lb_id"])
    return {k: sorted(v) for k, v in idx.items()}


def surface_candidates(tuples_path: Path | None = None) -> list[CandidatePair]:
    df = pd.read_parquet(tuples_path or (EXTRACTIONS / "tuples.parquet"))
    df = df[df["value"].map(_finite)].copy()
    df["value"] = df["value"].astype(float)

    # canonical display names
    import json as _json
    canon = _json.load(gzip.open(INDEX / "canon_tables.json.gz", "rt"))["entries"]

    def canon_name(cid: str, fallback: str) -> str:
        e = canon.get(str(cid))
        return e["canonical_name"] if e else str(fallback)

    comember = build_comembership_index()

    df["cell"] = list(zip(df["method_id"], df["dataset_id"], df["metric_id"]))
    pairs: list[CandidatePair] = []
    pair_n = 0
    for cell, g in df.groupby("cell", sort=True):
        if g["paper_id"].nunique() < 2:
            continue
        mid, did, met = (str(x) for x in cell)
        grade = identity_grade(mid, did, met)
        mdir = metric_direction(str(g["metric"].iloc[0]))
        m_can = canon_name(mid, g["method"].iloc[0])
        d_can = canon_name(did, g["dataset"].iloc[0])
        met_can = canon_name(met, g["metric"].iloc[0])
        lb_ids = comember.get((_norm(g["dataset"].iloc[0]), _norm(g["metric"].iloc[0])), [])
        split = split_for_dataset(did)
        # one representative per paper (RULING 2)
        reps: dict[str, object] = {}
        for pid, pg in g.groupby("paper_id", sort=True):
            reps[pid] = select_representative(list(pg.itertuples(index=False)))
        # cross-paper pairs with differing reconciled value
        for pa, pb in combinations(sorted(reps), 2):
            ra, rb = reps[pa], reps[pb]
            va, vb = float(ra.value), float(rb.value)
            xa, xb, adjusted = reconcile(va, vb)
            gap = abs(xa - xb)
            if gap <= _EPS:
                continue  # identical after reconciliation -> not a disagreement
            denom = max(abs(xa), abs(xb), _EPS)
            own_a, own_b = bool(ra.is_own_result), bool(rb.is_own_result)
            if own_a and own_b:
                ptype = "both_own"
            elif own_a != own_b:
                ptype = "own_vs_cited"
            else:
                ptype = "cited_vs_cited"
            pair_n += 1
            left, right = _cell_to_record(ra), _cell_to_record(rb)
            cp = CandidatePair(
                pair_id=f"cp{pair_n:06d}", method_id=mid, dataset_id=did, metric_id=met,
                method_canonical=m_can, dataset_canonical=d_can, metric_canonical=met_can,
                metric_direction=mdir, left=left, right=right,
                value_gap=round(gap, 6), rel_gap=round(gap / denom, 6),
                unit_consistent=not adjusted, identity_grade=grade,
                pair_type=ptype, task_family=(_norm(ra.task) or _norm(rb.task) or "unknown"),
                split=split, unit_scale_reconciled=adjusted,
                n_protocols_on_dataset_metric=len(lb_ids),
                comembership_lb_ids=lb_ids[:50],
            )
            pairs.append(cp)
    return pairs

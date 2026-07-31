# PROVENANCE: copied verbatim 2026-06-18 from paper3_completeqa/src/protocol/features.py
#   (old Chapter 3 (CompleteQA) protocol-cue engine, reused as the comparability-judge feature layer).
# The source repository is READ-ONLY; this is the working copy for paper3.2
#   (comparekg, Chapter 3 cross-paper result-cell disagreement census).
# Do not edit the source; edit this copy if behavior must change here.
"""Pair features for the protocol-equivalence classifier.

NO GOLD, NO LLM: operates only on extracted tuple fields. Shared by training
(src/protocol/pairs.py derives labels; this builds X) and inference
(src/protocol/infer.py). Both tuples in a pair already facet-match the same
(dataset, metric); the discriminative signal is therefore the protocol cues
that the facet match ignores — split, subdataset/parenthetical, exact metric
variant, task, unit, value magnitude, and source location.
"""

from __future__ import annotations

import math
import re

from rapidfuzz import fuzz

_PAREN = re.compile(r"[\(\[]([^)\]]*)[)\]]")
_WS = re.compile(r"\s+")


def _norm(s) -> str:
    return _WS.sub(" ", str(s or "").strip().lower())


def _paren(s: str) -> str:
    m = _PAREN.findall(s or "")
    return _norm(" ".join(m))


def _mag_bucket(v) -> float:
    try:
        v = abs(float(v))
    except (TypeError, ValueError):
        return -1.0
    if v == 0:
        return 0.0
    return float(round(math.log10(v)))


FEATURE_NAMES = [
    "dataset_sim", "dataset_paren_sim", "metric_sim", "split_match",
    "split_both_null", "split_sim", "task_sim", "unit_match",
    "same_paper", "mag_match", "source_block_sim", "quote_sim",
    "metric_exact", "dataset_exact",
    # descriptor-derived (0 when descriptors absent — back-compatible)
    "desc_benchmark_sim", "desc_split_match", "desc_setting_sim",
    "desc_metric_match", "desc_setting_both_known",
    # Option B structured-protocol-field features (0 when fields absent)
    "pf_key_match", "pf_split_match", "pf_setting_match", "pf_subtask_sim",
    "pf_metric_variant_match", "pf_setting_both_known", "pf_split_both_known",
]


def _pf_features(p1: dict | None, p2: dict | None) -> dict:
    if not p1 or not p2:
        return {k: 0.0 for k in ("pf_key_match", "pf_split_match", "pf_setting_match",
                                 "pf_subtask_sim", "pf_metric_variant_match",
                                 "pf_setting_both_known", "pf_split_both_known")}
    def n(d, k): return _norm(d.get(k))
    sp1, sp2 = n(p1, "split"), n(p2, "split")
    se1, se2 = n(p1, "setting"), n(p2, "setting")
    mv1, mv2 = n(p1, "metric_variant"), n(p2, "metric_variant")
    sp_known = sp1 not in ("", "unknown") and sp2 not in ("", "unknown")
    se_known = se1 not in ("", "unknown") and se2 not in ("", "unknown")
    key_match = (n(p1, "benchmark") == n(p2, "benchmark") and sp1 == sp2
                 and se1 == se2 and n(p1, "subtask") == n(p2, "subtask") and mv1 == mv2)
    return {
        "pf_key_match": float(key_match),
        "pf_split_match": float(sp1 == sp2 and sp_known),
        "pf_setting_match": float(se1 == se2 and se_known),
        "pf_subtask_sim": fuzz.token_set_ratio(n(p1, "subtask"), n(p2, "subtask")) / 100.0,
        "pf_metric_variant_match": float(mv1 == mv2 and mv1 not in ("", "none", "unknown")),
        "pf_setting_both_known": float(se_known),
        "pf_split_both_known": float(sp_known),
    }


def _desc_features(d1: dict | None, d2: dict | None) -> dict:
    if not d1 or not d2:
        return {"desc_benchmark_sim": 0.0, "desc_split_match": 0.0, "desc_setting_sim": 0.0,
                "desc_metric_match": 0.0, "desc_setting_both_known": 0.0}
    s1, s2 = _norm(d1.get("setting")), _norm(d2.get("setting"))
    sp1, sp2 = _norm(d1.get("split")), _norm(d2.get("split"))
    return {
        "desc_benchmark_sim": fuzz.token_set_ratio(_norm(d1.get("benchmark")), _norm(d2.get("benchmark"))) / 100.0,
        "desc_split_match": float(sp1 == sp2 and sp1 not in ("", "unknown")),
        "desc_setting_sim": fuzz.token_set_ratio(s1, s2) / 100.0,
        "desc_metric_match": float(_norm(d1.get("metric_norm")) == _norm(d2.get("metric_norm"))
                                   and _norm(d1.get("metric_norm")) not in ("", "unknown")),
        "desc_setting_both_known": float(s1 not in ("", "unknown") and s2 not in ("", "unknown")),
    }


def pair_features(t1: dict, t2: dict, dd1: dict | None = None, dd2: dict | None = None,
                  pf1: dict | None = None, pf2: dict | None = None) -> dict:
    d1, d2 = _norm(t1.get("dataset")), _norm(t2.get("dataset"))
    m1, m2 = _norm(t1.get("metric")), _norm(t2.get("metric"))
    s1, s2 = _norm(t1.get("split")), _norm(t2.get("split"))
    return {
        "dataset_sim": fuzz.token_set_ratio(d1, d2) / 100.0,
        "dataset_paren_sim": fuzz.token_set_ratio(_paren(t1.get("dataset")), _paren(t2.get("dataset"))) / 100.0,
        "metric_sim": fuzz.token_set_ratio(m1, m2) / 100.0,
        "split_match": float(s1 == s2 and bool(s1)),
        "split_both_null": float(not s1 and not s2),
        "split_sim": fuzz.token_set_ratio(s1, s2) / 100.0,
        "task_sim": fuzz.token_set_ratio(_norm(t1.get("task")), _norm(t2.get("task"))) / 100.0,
        "unit_match": float(_norm(t1.get("unit")) == _norm(t2.get("unit"))),
        "same_paper": float(t1.get("paper_id") == t2.get("paper_id")),
        "mag_match": float(_mag_bucket(t1.get("value")) == _mag_bucket(t2.get("value"))),
        "source_block_sim": fuzz.token_set_ratio(_norm(t1.get("source_block")), _norm(t2.get("source_block"))) / 100.0,
        "quote_sim": fuzz.token_set_ratio(_norm(t1.get("evidence_quote")), _norm(t2.get("evidence_quote"))) / 100.0,
        "metric_exact": float(m1 == m2 and bool(m1)),
        "dataset_exact": float(d1 == d2 and bool(d1)),
        **_desc_features(dd1, dd2),
        **_pf_features(pf1, pf2),
    }


def features_vector(t1: dict, t2: dict, d1: dict | None = None, d2: dict | None = None,
                    pf1: dict | None = None, pf2: dict | None = None) -> list[float]:
    f = pair_features(t1, t2, d1, d2, pf1, pf2)
    return [f[k] for k in FEATURE_NAMES]

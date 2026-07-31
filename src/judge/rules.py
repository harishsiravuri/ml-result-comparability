"""Deterministic rule signals for the comparability judge (NO LLM, NO curated gold).

These signals (a) are passed to the LLM as hints and (b) form the standalone rule labels
used for the auto-derivable cross-check against the human gold (Phase 4/5). They operate
only on the candidate record's extracted fields. Where a rule is confident it proposes a
cause; the LLM still makes the final call, and we record both.
"""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")

# canonical split vocabulary -> normalized split family
_SPLIT_VOCAB = {
    "test": "test", "testing": "test", "test set": "test",
    "val": "val", "valid": "val", "validation": "val", "dev": "val", "development": "val",
    "train": "train", "training": "train",
    "test-dev": "test-dev", "testdev": "test-dev", "test-std": "test-std",
    "minival": "val", "trainval": "trainval", "full": "full", "all": "full",
}

# metric-variant alias families: tokens that flip the variant of the SAME metric name
_VARIANT_TOKENS = {
    "micro", "macro", "weighted", "samples", "filtered", "raw", "constrained",
    "unconstrained", "top-1", "top-5", "top1", "top5", "overall", "mean", "per-class",
    "instance", "frame", "video", "@1", "@5", "@10", "@20", "@50", "@100",
}


def _norm(s) -> str:
    return _WS.sub(" ", str(s or "").strip().lower())


def _split_family(raw: str) -> str | None:
    n = _norm(raw)
    if not n:
        return None
    if n in _SPLIT_VOCAB:
        return _SPLIT_VOCAB[n]
    for k, v in _SPLIT_VOCAB.items():
        if k in n:
            return v
    return None


def split_signal(left_split: str, right_split: str) -> dict:
    a, b = _split_family(left_split), _split_family(right_split)
    known = a is not None and b is not None
    return {"left_split_family": a, "right_split_family": b,
            "both_known": known, "differs": bool(known and a != b)}


def metric_variant_signal(left_metric: str, right_metric: str) -> dict:
    la, lb = _norm(left_metric), _norm(right_metric)
    surface_differs = bool(la and lb and la != lb)
    ta = {t for t in re.findall(r"[a-z0-9@.\-]+", la)} & _VARIANT_TOKENS
    tb = {t for t in re.findall(r"[a-z0-9@.\-]+", lb)} & _VARIANT_TOKENS
    variant_token_differs = ta != tb and bool(ta or tb)
    return {"surface_differs": surface_differs,
            "variant_tokens_left": sorted(ta), "variant_tokens_right": sorted(tb),
            "variant_token_differs": variant_token_differs}


def extraction_risk_signal(pair: dict) -> dict:
    """Heuristic cues that a flagged disagreement might be an extraction artifact."""
    lv = pair["left"].get("critic_verdict")
    rv = pair["right"].get("critic_verdict")
    unsupported = (lv == "UNSUPPORTED") or (rv == "UNSUPPORTED")
    low_selfcons = (float(pair["left"].get("self_consistency") or 1.0) < 0.5) or \
                   (float(pair["right"].get("self_consistency") or 1.0) < 0.5)
    hash_only = pair.get("identity_grade") == "hash_only"
    extreme = float(pair.get("rel_gap") or 0.0) > 0.9     # near-total disagreement is suspicious
    return {"either_unsupported": bool(unsupported), "low_self_consistency": bool(low_selfcons),
            "hash_only_identity": bool(hash_only), "extreme_gap": bool(extreme)}


def rule_label(pair: dict) -> dict:
    """A standalone (LLM-free) cause guess + the raw signals.

    Used for the auto-derivable cross-check. Returns label None when no rule is confident.
    """
    sp = split_signal(pair["left"].get("split"), pair["right"].get("split"))
    mv = metric_variant_signal(pair["left"].get("metric"), pair["right"].get("metric"))
    ex = extraction_risk_signal(pair)
    label = None
    if sp["differs"]:
        label = "split"            # known split difference -> protocol_artifact/split
    elif mv["variant_token_differs"]:
        label = "metric_variant"   # a variant token flips -> protocol_artifact/metric_variant
    return {
        "rule_label": label,
        "split": sp, "metric_variant": mv, "extraction_risk": ex,
        "n_protocols_on_dataset_metric": pair.get("n_protocols_on_dataset_metric", 0),
    }

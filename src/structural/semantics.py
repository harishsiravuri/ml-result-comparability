# PROVENANCE: copied 2026-06-18 from paper2.3/src/fixkg/common/semantics.py (metric semantics), READ-ONLY source (Chapter 2.3 fixkg).
# Reused here as a STRUCTURAL FEATURE source for the Chapter 3 comparison-validity
# detector. Target = validity of a cross-paper COMPARISON, not single-fact correctness.
"""Metric semantics and numeric helpers (self-contained; provenance: ported from
paper2.2 src/paper2_2/{label/match.py, features/value_aware.py}, read-only source).
No gold value enters these helpers."""
from __future__ import annotations

import re

_BOUNDED_STRICT = {"accuracy", "acc", "f1", "f", "precision", "recall", "auc", "auroc",
                   "iou", "miou", "dice", "em", "map", "ap", "hits", "mrr", "ndcg",
                   "bleu", "rouge", "meteor", "spearman", "pearson", "kappa", "psnr", "ssim"}
_ERROR_TOKENS = {"error", "err", "wer", "cer", "per", "rmse", "mae", "mse", "loss",
                 "perplexity", "ppl", "fid", "nll", "eer", "der", "mad"}
_LOWER_TOKENS = _ERROR_TOKENS
_HIGHER_TOKENS = _BOUNDED_STRICT | {"score", "bleu", "rouge"}


def norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _toks(metric: str | None):
    return set(re.split(r"[^a-z0-9]+", norm(metric)))


def metric_type(metric: str | None) -> str:
    t = _toks(metric)
    if t & _ERROR_TOKENS:
        return "error"
    if t & _BOUNDED_STRICT:
        return "bounded"
    return "unknown"


def metric_direction(metric: str | None) -> str:
    """higher / lower / unknown (lower-is-better wins on conflict)."""
    t = _toks(metric)
    if t & _LOWER_TOKENS:
        return "lower"
    if t & _HIGHER_TOKENS:
        return "higher"
    return "unknown"


def value_close(v1: float, v2: float, rel: float = 0.01) -> bool:
    """Locked tolerance: scale=max(|a|,|b|); abs_tol=0.5 if scale>1 else 0.005; match if
    |a-b|<=max(abs_tol, rel*scale); plus x100 percent/fraction rescale."""
    def ok(a, b):
        s = max(abs(a), abs(b)); at = 0.5 if s > 1 else 0.005
        return abs(a - b) <= max(at, rel * s)
    return ok(v1, v2) or ok(v1 * 100, v2) or ok(v1 * 0.01, v2)


def bounded_violation(metric: str | None, value: float) -> bool:
    """Bounded metric value outside [0, 100] (percent or fraction scale both <=100)."""
    if metric_type(metric) != "bounded":
        return False
    return value < 0 or value > 100.0


def error_violation(metric: str | None, value: float) -> bool:
    return metric_type(metric) == "error" and value < 0

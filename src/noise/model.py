"""Preregistered noise model (AMENDMENT A, 2026-06-18). Deterministic, NO LLM, $0.

ONE unified beyond-noise rule:

  BEYOND NOISE  iff  |v1 - v2|  >  k * sqrt(sigma_1^2 + sigma_2^2)

where each sigma_i is the side's dispersion: the REPORTED dispersion (parsed from the
evidence span) if present, else a calibrated DEFAULT. k = 2 (about a 95% interval under
normality). The rule is metric-type-aware because a relative-gap rule misbehaves near a
bounded ceiling or floor and most leaderboard metrics are bounded:

  - BOUNDED metrics (accuracy, F1, mIoU, MRR, Hits@k, BLEU, ...): work in absolute
    POINTS on a 0-100 scale. Default sigma = SIGMA_DEFAULT_BOUNDED_PTS points.
  - UNBOUNDED ratio metrics (perplexity, FID, MSE, latency, Elo, ...): work in relative
    terms. Default sigma_i = SIGMA_DEFAULT_REL * |v_i|.
  - UNKNOWN range: require BOTH views to call beyond-noise (the more conservative rule).

Calibration (from 1,618 reported "X +/- S" spans; see scripts noise calibration):
  reported relative SD: median 0.0078, p75 0.0210, p90 0.0495.
  reported normalized absolute SD (points): median 0.50, p75 1.13, p90 2.50.
  SIGMA_DEFAULT_REL    = 0.021  (p75 relative SD). Emergent all-defaulted unbounded
                         threshold k*sqrt(2)*0.021 = 0.0594 relative.
  SIGMA_DEFAULT_BOUNDED_PTS = 0.71  (set so the all-defaulted bounded threshold
                         k*sqrt(2)*0.71 = 2.0 normalized points, the level anchored at
                         sign-off; it sits between the median (-> 1.41 pt threshold) and
                         the p75 (-> 3.19 pt threshold) of the reported absolute-SD
                         distribution, honoring "under-call rather than over-call").

We report the share of beyond-noise calls made on REPORTED vs DEFAULTED dispersion (it
bounds reliability), and a prevalence-versus-threshold sensitivity curve.

The Paper 2.2 trust score (soft dependency) will weight the per-cell dispersion when
available; the interface is left open (decide() accepts optional trust).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

K = 2.0
SIGMA_DEFAULT_REL = 0.021
SIGMA_DEFAULT_BOUNDED_PTS = 0.71
# sensitivity grid (relative-equivalent) reported alongside the primary
SENSITIVITY_REL_GRID = [0.02, 0.042, 0.0594, 0.10]

_TOKEN_SPLIT = re.compile(r"[^a-z0-9@#^.-]+")
_PM = re.compile(r"([0-9]+\.?[0-9]*)\s*(?:±|\+/-|±)\s*([0-9]+\.?[0-9]*)")

# Metric-range families. Unbounded takes precedence on conflict.
_UNBOUNDED = {
    "ppl", "perplexity", "fid", "kid", "nll", "loss", "mse", "mae", "rmse", "rmsle",
    "rms", "flops", "params", "latency", "runtime", "time", "elo", "regret", "epe",
    "mcd", "fvd", "swd", "l1", "l2", "distance", "chamfer", "bpc", "bpd", "ms", "fps",
}
_BOUNDED = {
    "accuracy", "acc", "f1", "f-score", "fscore", "precision", "recall", "auc", "auroc",
    "auprc", "ap", "map", "ap50", "ap75", "iou", "miou", "dice", "bleu", "rouge",
    "meteor", "cider", "spice", "ssim", "mrr", "ndcg", "hits", "hit", "em", "uas", "las",
    "mcc", "matthews", "spearman", "pearson", "kendall", "success", "win", "top-1",
    "top-5", "top1", "top5", "wer", "cer", "per", "ter", "err", "error", "rate",
    "psnr",  # treat as bounded-ish (dB but practically ranged); robustness view covers it
}


def classify_metric_range(name: str) -> str:
    """'bounded' | 'unbounded' | 'unknown' from the metric name alone (no gold).

    Precedence: an unbounded token (perplexity, FID, MSE, latency, Elo, ...) wins, since
    such metrics are unambiguous; then a bounded token (accuracy, F1, mIoU, ...); then a
    few unbounded phrase cues; else unknown (handled by the conservative AND rule).
    """
    n = (name or "").strip().lower()
    toks = {t for t in _TOKEN_SPLIT.split(n) if t}
    if toks & _UNBOUNDED:
        return "unbounded"
    if toks & _BOUNDED:
        return "bounded"
    if any(p in n for p in ("perplex", "distance", "wasserstein", "chamfer", "divergence")):
        return "unbounded"
    return "unknown"


def parse_reported_sd(evidence_quote: str, value: float) -> float | None:
    """Absolute SD in the value's own units, if the span carries 'X +/- S' with X~value."""
    if not evidence_quote:
        return None
    best = None
    for m in _PM.finditer(evidence_quote):
        base, sd = float(m.group(1)), float(m.group(2))
        if base <= 0 or sd < 0 or sd >= base:
            continue
        # accept if base matches the value at native scale or x100 scale
        for scale in (1.0, 0.01, 100.0):
            if abs(base * scale - value) <= max(0.05 * abs(value), 0.05):
                cand = sd * scale
                if best is None or cand < best:
                    best = cand
    return best


@dataclass
class NoiseDecision:
    beyond_noise: bool
    range_type: str            # bounded | unbounded | unknown
    threshold: float           # in decision units (points if bounded, value units if unbounded)
    gap: float                 # in the same decision units
    sigma_left: float
    sigma_right: float
    sd_source_left: str        # reported | defaulted
    sd_source_right: str
    decision_units: str        # "points_0_100" | "value_units" | "both(AND)"


def _to_points(v1: float, v2: float) -> tuple[float, float]:
    if max(abs(v1), abs(v2)) <= 1.5:
        return v1 * 100.0, v2 * 100.0
    return v1, v2


def _decide_bounded(v1, v2, sd1, sd2) -> tuple[bool, float, float, float, float]:
    p1, p2 = _to_points(v1, v2)
    # scale reported SDs to points if values were scaled
    scaled = max(abs(v1), abs(v2)) <= 1.5
    s1 = (sd1 * 100.0 if (sd1 is not None and scaled) else sd1) if sd1 is not None else SIGMA_DEFAULT_BOUNDED_PTS
    s2 = (sd2 * 100.0 if (sd2 is not None and scaled) else sd2) if sd2 is not None else SIGMA_DEFAULT_BOUNDED_PTS
    thr = K * math.sqrt(s1 * s1 + s2 * s2)
    return abs(p1 - p2) > thr, thr, abs(p1 - p2), s1, s2


def _decide_unbounded(v1, v2, sd1, sd2) -> tuple[bool, float, float, float, float]:
    s1 = sd1 if sd1 is not None else SIGMA_DEFAULT_REL * abs(v1)
    s2 = sd2 if sd2 is not None else SIGMA_DEFAULT_REL * abs(v2)
    thr = K * math.sqrt(s1 * s1 + s2 * s2)
    return abs(v1 - v2) > thr, thr, abs(v1 - v2), s1, s2


def decide(v1: float, v2: float, metric_name: str, quote1: str = "", quote2: str = "",
           trust_left: float | None = None, trust_right: float | None = None) -> NoiseDecision:
    """Primary beyond-noise decision for one (already unit-reconciled) pair."""
    rng = classify_metric_range(metric_name)
    sd1, sd2 = parse_reported_sd(quote1, v1), parse_reported_sd(quote2, v2)
    src1 = "reported" if sd1 is not None else "defaulted"
    src2 = "reported" if sd2 is not None else "defaulted"
    if rng == "bounded":
        b, thr, gap, s1, s2 = _decide_bounded(v1, v2, sd1, sd2)
        return NoiseDecision(b, rng, thr, gap, s1, s2, src1, src2, "points_0_100")
    if rng == "unbounded":
        b, thr, gap, s1, s2 = _decide_unbounded(v1, v2, sd1, sd2)
        return NoiseDecision(b, rng, thr, gap, s1, s2, src1, src2, "value_units")
    # unknown: more conservative -> require BOTH views to fire
    bb, tb, gb, _, _ = _decide_bounded(v1, v2, sd1, sd2)
    bu, tu, gu, s1, s2 = _decide_unbounded(v1, v2, sd1, sd2)
    return NoiseDecision(bb and bu, rng, tb, gb, s1, s2, src1, src2, "both(AND)")

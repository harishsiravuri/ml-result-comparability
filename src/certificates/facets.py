"""Protocol-facet extraction and relation logic (framework doc sections 2 and 4).

A facet is OBSERVED (a known normalized value) or MISSING (not recoverable from the paper
text by the pipeline). Missingness is first-class: neither "same" nor "different".

The three facets the pipeline actually extracts, and therefore the only ones that can carry
an observed relation, are: split, metric_surface, unit. The remaining inventory dimensions
(evaluation_setting, training_data, preprocessing, dataset_version, aggregation_scope,
compute) are recorded as MISSING for every pair, which is the honest state of the corpus.
"""

from __future__ import annotations

import re

from census.surface import _norm
from judge.rules import _VARIANT_TOKENS, _split_family

# Facets the pipeline can observe (the must-agree set for the default/unconditional view).
INSPECTED_FACETS = ["split", "metric_surface", "unit"]

# Inventory dimensions the pipeline cannot recover at scale (framework doc section 4).
NEVER_OBSERVED_DIMENSIONS = [
    "evaluation_setting", "training_data", "preprocessing",
    "dataset_version", "aggregation_scope", "compute",
]

# Metric-surface tokens that denote a cutoff or aggregation of a SHARED base metric.
# A difference in these is "partially comparable" (comparable at the shared operating
# point), not a flat incompatibility (framework doc section 5).
_CUTOFF_AGG_TOKENS = {
    "micro", "macro", "weighted", "samples", "filtered", "raw",
    "@1", "@3", "@5", "@10", "@20", "@50", "@100", "top-1", "top-5", "top1", "top5",
    "per-class", "overall", "instance", "frame", "video",
}

# The variant vocabulary the whole pipeline recognizes. Reuses the FROZEN judge's token set
# (judge.rules._VARIANT_TOKENS) rather than defining a new one, so the facet layer and the
# judge agree on what counts as a variant.
_VARIANT_VOCAB = _VARIANT_TOKENS | _CUTOFF_AGG_TOKENS


# A trailing percent annotation on the metric surface is a UNIT statement, not a metric
# variant: "accuracy (%)" and "accuracy" are the same metric surface. Left unnormalized it
# contaminates any metric-variant analysis (26 of the 344 differing-surface census pairs
# differ ONLY by this annotation) and produces spurious metric_surface incompatibilities.
_UNIT_ANNOT_RE = re.compile(
    r"\s*(?:\(\s*(?:%|percent|percentage|in\s*%)\s*\)|\[\s*%\s*\]|%)\s*$")


def strip_unit_annotation(surface: str | None) -> tuple[str | None, str | None]:
    """Split a trailing percent annotation off the metric surface.

    Returns (base_surface, unit_annotation | None). A surface that is ONLY the annotation
    is left intact, since stripping it would erase the metric.
    """
    if not surface:
        return surface, None
    m = _UNIT_ANNOT_RE.search(surface)
    if not m:
        return surface, None
    base = surface[: m.start()].strip()
    if not base:
        return surface, None
    return base, "%"


def side_protocol(side: dict) -> dict:
    """Observed protocol signature of one side: facet -> normalized value or None (missing).

    The percent annotation is moved from the metric surface to the unit facet. It is only
    ADOPTED as the unit when no unit is separately observed, so an explicitly recorded unit
    always wins.
    """
    sp = _split_family(side.get("split"))
    ms = _norm(side.get("metric")) or None
    un = _norm(side.get("unit")) or None
    ms, annot = strip_unit_annotation(ms)
    if annot and not un:
        un = annot
    return {"split": sp or None, "metric_surface": ms or None, "unit": un}


def relation(a, b) -> str:
    if a is None or b is None:
        return "missing"
    return "observed-same" if a == b else "observed-different"


_ALIAS_NORM = {"facet": "metric_surface", "op": "metric_surface_alias_normalization",
               "side": "auto"}


def variant_signature(surface: str | None) -> frozenset:
    """The variant-bearing tokens of a metric surface (micro/macro, @k, filtered/raw, ...).

    Empty means the surface states NO variant, which is not the same as stating a default.
    """
    return frozenset(_tokens(surface) & _VARIANT_VOCAB)


def metric_surface_relation(a: str | None, b: str | None) -> tuple[str, dict | None]:
    """Relation for metric_surface, plus any normalization it required.

    Both sides of a pair already share a canonical metric_id, so the metric NAME carries no
    comparability signal: the surface's only comparability-relevant content is the VARIANT it
    states. Hence:

      - equal signatures, different spelling ("acc" vs "accuracy", "miou" vs "mean iou")
        -> observed-same under a recorded alias normalization, NOT a protocol difference;
      - both sides state a variant and the variants differ -> observed-different;
      - exactly one side states a variant ("overall accuracy" vs "accuracy") -> MISSING. The
        silent side may well use the same variant; saying less is not saying something else.
    """
    if a is None or b is None:
        return "missing", None
    if a == b:
        return "observed-same", None
    sa, sb = variant_signature(a), variant_signature(b)
    if sa and sb:
        if sa != sb:
            return "observed-different", None
        return "observed-same", dict(_ALIAS_NORM)
    if bool(sa) != bool(sb):
        return "missing", None
    return "observed-same", dict(_ALIAS_NORM)


def per_facet(left_proto: dict, right_proto: dict) -> list[dict]:
    rows = []
    for f in INSPECTED_FACETS:
        lv, rv = left_proto.get(f), right_proto.get(f)
        rel = (metric_surface_relation(lv, rv)[0] if f == "metric_surface"
               else relation(lv, rv))
        rows.append({"facet": f, "left": lv, "right": rv, "relation": rel})
    return rows


def surface_normalizations(left_proto: dict, right_proto: dict) -> list[dict]:
    """Normalizations the facet layer itself applied (currently the metric-surface alias)."""
    _, norm = metric_surface_relation(left_proto.get("metric_surface"),
                                      right_proto.get("metric_surface"))
    return [norm] if norm else []


def _tokens(surface: str | None) -> set[str]:
    """Word tokens, with an attached cutoff split off so `hits@10` yields {hits, @10}."""
    if not surface:
        return set()
    out = set()
    for t in re.split(r"[^a-z0-9@.\-]+", surface.lower()):
        if not t:
            continue
        m = re.fullmatch(r"(.*?)(@\d+)", t)
        if m and m.group(1):
            out.update({m.group(1), m.group(2)})
        else:
            out.add(t)
    return out


def is_cutoff_or_aggregation_difference(left_ms: str | None, right_ms: str | None) -> bool:
    """True when the two metric surfaces differ only by a cutoff/aggregation of a shared base
    (for example micro versus macro F1, or Recall@20 versus Recall@50)."""
    if not left_ms or not right_ms or left_ms == right_ms:
        return False
    lt, rt = _tokens(left_ms), _tokens(right_ms)
    lv, rv = lt & _CUTOFF_AGG_TOKENS, rt & _CUTOFF_AGG_TOKENS
    if lv == rv:
        return False           # the difference is not in the cutoff/aggregation tokens
    return bool((lt - lv) & (rt - rv)) or bool(lv and rv)  # a shared base remains

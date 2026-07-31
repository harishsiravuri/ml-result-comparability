"""Item 3: deterministic remap of the resource's statuses and census causes onto the six
advisor typed outcomes (framework doc section 5).

HONESTY RULE (framework doc section 5, forced by the partition audit): a pair whose OBSERVED
protocol facets all agree routes to "comparable under assumptions" by DEFAULT, not to
"directly comparable", because hidden facets are frequently unequal (the audit found 13 of 15
same-observed-protocol pairs still incomparable to the human). "Directly comparable" is
reserved for the rare case where no inventory facet is missing, which the corpus essentially
never supplies.

Census causes: within_noise and genuine_conflict presuppose comparability and INHERIT the
pair's facet-derived comparability type; extraction_artifact is EXCLUDED as not a genuine
pair (reported as a separate bucket, not one of the six).
"""

from __future__ import annotations

from certificates.facets import (
    NEVER_OBSERVED_DIMENSIONS,
    is_cutoff_or_aggregation_difference,
)

SIX_OUTCOMES = [
    "directly_comparable",
    "comparable_after_deterministic_normalization",
    "comparable_under_assumptions",
    "partially_comparable",
    "incompatible",
    "unknown",
]
EXCLUDED = "excluded_not_a_genuine_pair"


def typed_outcome(per_facet_rows: list[dict], normalizations: list[dict],
                  missing_dimensions: list[str], judge_cause: str | None) -> dict:
    """Return {typed_outcome, rule, requires}. Deterministic; first matching rule wins."""
    rel = {r["facet"]: r["relation"] for r in per_facet_rows}
    val = {r["facet"]: (r["left"], r["right"]) for r in per_facet_rows}

    # 0. extraction artifact is not a genuine pair (framework doc section 5).
    if judge_cause == "extraction_artifact":
        return {"typed_outcome": EXCLUDED,
                "rule": "cause==extraction_artifact -> excluded (not a genuine pair)",
                "requires": "a corrected extraction or identity match"}

    # 1. an observed split difference is a hard cross-protocol incompatibility.
    if rel.get("split") == "observed-different":
        return {"typed_outcome": "incompatible",
                "rule": "split observed-and-different",
                "requires": "an observed conflicting facet (split)"}

    # 2. a metric-surface difference that is a cutoff/aggregation of a shared base is
    #    comparable only at the shared operating point.
    if rel.get("metric_surface") == "observed-different":
        lms, rms = val["metric_surface"]
        if is_cutoff_or_aggregation_difference(lms, rms):
            return {"typed_outcome": "partially_comparable",
                    "rule": "metric_surface differs by a cutoff/aggregation of a shared base",
                    "requires": "a shared sub-metric or operating point"}
        return {"typed_outcome": "incompatible",
                "rule": "metric_surface observed-and-different (not a shared-base cutoff)",
                "requires": "an observed conflicting facet (metric surface)"}

    # 3. the only differences are ones a deterministic normalization reconciles (a
    #    percent-vs-fraction unit rescale, or a metric-surface alias with an identical variant
    #    signature). This requires every OTHER inspected facet to be observed-same: a
    #    normalization cannot certify comparability while a must-agree facet (for example the
    #    split) is still missing.
    normalized = {n["facet"] for n in normalizations}
    others = [f for f in rel if f not in normalized]
    if normalizations and all(rel.get(f) == "observed-same" for f in others):
        return {"typed_outcome": "comparable_after_deterministic_normalization",
                "rule": "differs only by a reversible normalization (unit rescale and/or "
                        "metric-surface alias) and all other inspected facets observed-same",
                "requires": "a known, reversible normalization"}
    if rel.get("unit") == "observed-different" and not normalizations:
        return {"typed_outcome": "incompatible",
                "rule": "unit observed-and-different with no applicable normalization",
                "requires": "an observed conflicting facet (unit)"}

    # 4/5. no observed difference remains: same observed protocol.
    if all(rel.get(f) == "observed-same" for f in rel):
        if not missing_dimensions:
            return {"typed_outcome": "directly_comparable",
                    "rule": "all inspected facets observed-same AND no inventory facet missing",
                    "requires": "complete facets, no normalization"}
        return {"typed_outcome": "comparable_under_assumptions",
                "rule": "all OBSERVED facets agree but >=1 inventory facet is missing "
                        "(default per the honesty rule)",
                "requires": "an explicit stated assumption that the unobserved facets are equal"}

    # 6. at least one must-agree facet is missing and none is observed-different.
    return {"typed_outcome": "unknown",
            "rule": ">=1 inspected facet missing on a side; none observed-different",
            "requires": "more protocol detail than the paper text provides"}


def missing_dimensions_for(per_facet_rows: list[dict]) -> list[str]:
    """Inventory dimensions that are missing for this pair: the never-observed dimensions plus
    any inspected facet whose relation is 'missing'."""
    miss = [r["facet"] for r in per_facet_rows if r["relation"] == "missing"]
    return miss + list(NEVER_OBSERVED_DIMENSIONS)

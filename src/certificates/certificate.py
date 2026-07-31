"""Item 2: build a comparability certificate for one comparison (framework doc section 6).

Two status fields, deliberately distinct:
  status         - what the RESOURCE reports (the judge cause where a judge trace exists,
                   otherwise the deterministic facet decision);
  implied_status - what per_facet plus normalizations_applied alone imply.
Auditability requires status == implied_status; `self_consistent` records whether it holds.

Where the judge names a dimension that the facet extractor cannot observe (typically
evaluation_setting), the certificate records an ASSERTED per_facet row so the status stays
reconstructable, marked source="asserted_by_judge" and carrying no facet values.
Pointer-only provenance throughout: no excerpt text.
"""

from __future__ import annotations

from certificates.facets import (
    INSPECTED_FACETS,
    per_facet,
    side_protocol,
    surface_normalizations,
)
from certificates.typed_outcomes import missing_dimensions_for, typed_outcome

CAUSE_TO_STATUS = {
    "split": "incompatible",
    "metric_variant": "incompatible",
    "evaluation_setting": "incompatible",
    "genuine_conflict": "comparable",
    "citation_reporting_discrepancy": "comparable",
    "extraction_artifact": "excluded",
    "undetermined": "unknown",
}
CAUSE_TO_FACET = {"split": "split", "metric_variant": "metric_surface",
                  "evaluation_setting": "evaluation_setting"}


def implied_status(rows: list[dict], normalizations: list[dict]) -> str:
    """Status implied by the recorded facets alone."""
    normalized_facets = {n["facet"] for n in normalizations}
    diff = [r for r in rows
            if r["relation"] == "observed-different" and r["facet"] not in normalized_facets]
    if diff:
        return "incompatible"
    extracted = [r for r in rows if r["source"] == "extracted"]
    if extracted and all(r["relation"] == "observed-same"
                         or (r["facet"] in normalized_facets) for r in extracted):
        return "comparable"
    return "unknown"


def build_certificate(cert_id, source, cell, left_rec, right_rec, left_proto, right_proto,
                      normalizations, judge_cause, uncertainty, query=None):
    """left_rec/right_rec: {arxiv_id, source_location, value}. *_proto: facet -> value|None."""
    rows = [dict(r, source="extracted") for r in per_facet(left_proto, right_proto)]
    # the facet layer may itself have normalized a metric-surface alias; record it so the
    # status stays reconstructable from per_facet + normalizations_applied
    normalizations = list(normalizations) + surface_normalizations(left_proto, right_proto)

    # record a judge-asserted dimension when the judge names one the facets do not show
    if judge_cause in CAUSE_TO_FACET:
        f = CAUSE_TO_FACET[judge_cause]
        already = any(r["facet"] == f and r["relation"] == "observed-different" for r in rows)
        if not already:
            rows.append({"facet": f, "left": None, "right": None,
                         "relation": "observed-different", "source": "asserted_by_judge"})

    missing = missing_dimensions_for([r for r in rows if r["source"] == "extracted"])
    # a dimension the judge asserted as different is no longer "missing"
    asserted = {r["facet"] for r in rows if r["source"] == "asserted_by_judge"}
    missing = [d for d in missing if d not in asserted]

    t = typed_outcome([r for r in rows if r["source"] == "extracted"] +
                      [r for r in rows if r["source"] == "asserted_by_judge"],
                      normalizations, missing, judge_cause)

    imp = "excluded" if judge_cause == "extraction_artifact" else implied_status(rows, normalizations)
    rep = CAUSE_TO_STATUS.get(judge_cause) if judge_cause else imp

    compatible = [r["facet"] for r in rows if r["relation"] == "observed-same"]
    incompatible = [r["facet"] for r in rows if r["relation"] == "observed-different"]

    return {
        "certificate_version": "1.0",
        "certificate_id": cert_id,
        "source": source,
        "query": query or {"template": "unconditional", "must_agree_facets": list(INSPECTED_FACETS)},
        "cell": cell,
        "left": {**left_rec, "protocol": left_proto},
        "right": {**right_rec, "protocol": right_proto},
        "inspected_facets": list(INSPECTED_FACETS),
        "per_facet": rows,
        "normalizations_applied": normalizations,
        "compatible_dimensions": compatible,
        "incompatible_dimensions": incompatible,
        "missing_dimensions": missing,
        "status": rep,
        "implied_status": imp,
        "self_consistent": rep == imp,
        "typed_outcome": t["typed_outcome"],
        "typed_outcome_rule": t["rule"],
        "typed_outcome_requires": t["requires"],
        "judge_cause": judge_cause,
        "uncertainty": uncertainty,
        "provenance_only": True,
        "evidence_text_redistributed": False,
    }

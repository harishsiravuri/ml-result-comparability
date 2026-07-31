"""Canonical registry of query-conditioned comparability templates (framework doc section 3).

This module is the single source of truth for the template -> required-condition mapping. The
released artifact `schemas/query_templates.json` is emitted from it by
`scripts/emit_query_templates.py`, the pilot `scripts/query_conditioned_pilot.py` imports the
facet lists from it, and the certificate schema's `query.template` enum is kept in sync (a test
asserts all three agree). Deterministic; no model, no gold.

Status semantics are shared across templates and adopt the CQA reading: a query-induced answer
is `robust` when it holds in every valid comparison set (the consistent answer over all
repairs), `conditional` when it holds in some valid sets but not all, and `insufficient` when
no valid comparison set is large enough to answer.
"""

from __future__ import annotations

SCHEMA_VERSION = "1.0.0"

STATUS_SEMANTICS = {
    "robust": "The query answer holds in every valid comparison set (the CQA consistent "
              "answer over all repairs).",
    "conditional": "The query answer holds in some valid comparison sets but not all; it "
                   "depends on which protocol you condition on.",
    "insufficient": "No valid comparison set is large enough to answer (the missing-facet "
                    "mass dominates, or the quantity is untestable on this leaderboard).",
}

# Order matters only for display. must_agree_facets is the set of protocol facets on which
# records must agree to share a valid comparison set. restrictions are additional filters
# applied before grouping. derived_from names a template a verification query builds on.
TEMPLATES = [
    {
        "name": "Q_any",
        "question": "Which method reports the best value under any protocol?",
        "must_agree_facets": [],
        "restrictions": [],
        "answer": "argmax value (the naive leaderboard winner)",
        "substantive": False,
        "note": "Robust on every leaderboard by construction: with no must-agree facet there "
                "is exactly one comparison set, so the naive query can never report "
                "uncertainty. This is the failure mode the framing targets, not a finding.",
    },
    {
        "name": "Q_official",
        "question": "Which method is best under the official (test) protocol?",
        "must_agree_facets": ["metric_surface", "unit"],
        "restrictions": [{"facet": "split", "op": "equals", "value": "test"}],
        "answer": "argmax value restricted to the official protocol",
        "substantive": True,
        "note": "Records not on the test split are excluded from every valid comparison set.",
    },
    {
        "name": "Q_trend",
        "question": "How does the best justified value evolve over publication year?",
        "must_agree_facets": ["split", "metric_surface", "unit"],
        "restrictions": [{"op": "partition_by", "key": "publication_year"}],
        "answer": "a per-year sequence of argmax winners",
        "substantive": True,
        "note": "Publication year is taken from the arXiv identifier prefix, so it is the "
                "preprint year.",
    },
    {
        "name": "Q_threshold",
        "question": "Does the official champion's lead over the runner-up survive protocol "
                    "conditioning?",
        "must_agree_facets": ["split", "metric_surface", "unit"],
        "restrictions": [
            {"op": "reference_from", "template": "Q_official",
             "detail": "champion M* and threshold tau (the best competitor value) are fixed "
                       "from the principal Q_official comparison set"},
            {"op": "test_over",
             "detail": "full-protocol groups {split, metric_surface, unit} in which M* is "
                       "measured against at least one other method"}],
        "answer": "whether M* strictly exceeds tau in every full-protocol group that measures "
                  "it against a peer",
        "substantive": True,
        "derived_from": "Q_official",
        "note": "The reference threshold and champion come from the official protocol; "
                "robustness is tested by whether that champion still clears the threshold "
                "under every other protocol in which it is measured against a peer, which is "
                "the faithful CQA reading of a margin query and avoids the vacuous case where "
                "the runner-up is trivially below the threshold within its own set. A "
                "leaderboard is insufficient when no principal official set exists or M* is "
                "measured against a peer under only one protocol.",
    },
]

TEMPLATE_NAMES = [t["name"] for t in TEMPLATES]
# `unconditional` is the default certificate view (all inspected facets must agree); it is not
# a pilot query but is a valid certificate template, so the enum is the pilot names plus it.
CERTIFICATE_TEMPLATE_ENUM = TEMPLATE_NAMES + ["unconditional"]


def must_agree(name: str) -> list[str]:
    for t in TEMPLATES:
        if t["name"] == name:
            return list(t["must_agree_facets"])
    raise KeyError(name)


def as_artifact() -> dict:
    """The exact content of schemas/query_templates.json."""
    return {
        "schema_version": SCHEMA_VERSION,
        "description": "Versioned registry of query-conditioned comparability templates "
                       "(framework document section 3). Each template maps to the protocol "
                       "facets records must agree on to share a valid comparison set, plus any "
                       "restrictions. Status semantics are shared and follow consistent query "
                       "answering over inconsistent databases.",
        "status_semantics": STATUS_SEMANTICS,
        "certificate_template_enum": CERTIFICATE_TEMPLATE_ENUM,
        "templates": TEMPLATES,
    }

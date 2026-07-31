"""Schemas for the cross-paper result-cell disagreement census.

This module is the single source of truth for the released-dataset row shape and
its PROVENANCE invariant: every flagged instance must be traceable to its two (or
more) source papers and to a source span in each. tests/test_provenance.py enforces
the invariant; src/census, src/noise, and src/judge produce rows that satisfy it.

Three record levels (each a superset of the prior):
  1. CandidateCell    - one extracted result-cell tuple (a paper's reported value
                        for a canonical (method, dataset, metric) identity, with its
                        source span). Built in Phase 1.
  2. CandidatePair    - two CandidateCells from DIFFERENT papers sharing the canonical
                        identity, with a differing value. The unit of disagreement.
                        Built in Phase 1.
  3. AttributedRow    - a CandidatePair that survived the noise model (Phase 2) and
                        received a cause from the comparability judge (Phase 3), plus
                        the human label where annotated (Phase 4/5). This IS the
                        released dataset row (Phase 6).

NO-LEAKAGE NOTE: the per-side `value` here is the PAPER-REPORTED value (the object of
study, extracted from the paper's own text). It is NOT the Papers-with-Code curated
"gold" value. The judge sees reported values and cues; it never sees the PwC canonical
value (see tests/test_no_leakage.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# --- cause taxonomy (FROZEN in PREREGISTRATION.md 06267c0, Section 4) ---
# Leaf labels the judge assigns. Top level = protocol_artifact{split/metric_variant/
# evaluation_setting} | citation_reporting_discrepancy | genuine_conflict |
# extraction_artifact(reject). "undetermined" = judge could not attribute (measured residual).
CAUSE_LABELS = (
    "split",                          # protocol_artifact / split
    "metric_variant",                 # protocol_artifact / metric_variant
    "evaluation_setting",             # protocol_artifact / evaluation_setting
    "citation_reporting_discrepancy",
    "genuine_conflict",
    "extraction_artifact",            # reject: not a real cross-paper disagreement
    "undetermined",                   # judge could not attribute (residual)
)
PRIMARY_CAUSES = ("split", "metric_variant")  # the chapter's validated core
PROTOCOL_SUBTYPES = ("split", "metric_variant", "evaluation_setting")


def top_level_cause(leaf: str) -> str:
    """Map a leaf cause to its top-level bucket (for the 4-class macro-F1 bar)."""
    if leaf in PROTOCOL_SUBTYPES:
        return "protocol_artifact"
    if leaf in ("citation_reporting_discrepancy", "genuine_conflict", "extraction_artifact"):
        return leaf
    return "undetermined"


@dataclass
class CandidateCell:
    """One paper's reported value for a canonical cell, with its source span."""

    paper_id: str
    method_id: str
    dataset_id: str
    metric_id: str
    method: str            # raw/surface method name as extracted
    dataset: str
    metric: str
    value: float
    unit: str | None
    split: str | None
    task: str | None
    is_own_result: bool
    evidence_quote: str    # the source span (provenance)
    source_block: str      # where in the paper (section/table) the span came from
    quote_verified: bool
    self_consistency: float | None
    critic_verdict: str | None

    def has_span(self) -> bool:
        return bool((self.evidence_quote or "").strip() or (self.source_block or "").strip())


@dataclass
class CandidatePair:
    """Two CandidateCells (different papers, same canonical identity, differing value)."""

    pair_id: str
    method_id: str
    dataset_id: str
    metric_id: str
    method_canonical: str
    dataset_canonical: str
    metric_canonical: str
    metric_direction: str          # higher | lower | unknown
    left: CandidateCell
    right: CandidateCell
    value_gap: float               # |left.value - right.value| in the cells' own units
    rel_gap: float                 # value_gap / max(|v|, eps)
    unit_consistent: bool          # both units compatible (no x100 scale mismatch)
    identity_grade: str            # all_pwc | partial_pwc | hash_only  (coverage feature)
    # --- Phase 1 stratification / coverage fields ---
    pair_type: str = "unknown"     # both_own | own_vs_cited | cited_vs_cited (RULING 1)
    task_family: str = "unknown"   # for the cross-task-family breakdown (RULING 7)
    split: str = "unassigned"      # dev | test (frozen, dataset-level)
    unit_scale_reconciled: bool = False           # a ~100x percent/fraction artifact was reconciled
    n_protocols_on_dataset_metric: int = 0        # # distinct PwC leaderboards on this dataset+metric
    comembership_lb_ids: list[str] = field(default_factory=list)  # shared PwC leaderboards (a feature, NOT a value)
    trust_left: float | None = None    # optional paper2.2 per-cell trust (soft dep)
    trust_right: float | None = None


@dataclass
class AttributedRow:
    """Released dataset row: a CandidatePair + noise decision + cause + provenance."""

    pair: CandidatePair
    # noise model (Phase 2)
    beyond_noise: bool
    noise_evidence: dict[str, Any]          # {source: reported|defaulted, sigma, threshold, ...}
    # comparability judge (Phase 3)
    cause: str                              # one of CAUSE_LABELS
    judge_rationale: str
    judge_backbones: list[str]              # >= 1; agreement reported across them
    judge_per_backbone: dict[str, str] = field(default_factory=dict)  # backbone -> cause
    rule_signals: dict[str, Any] = field(default_factory=dict)        # deterministic-rule hits
    # human gold (Phase 4/5), optional
    author_label_inconsistent: bool | None = None
    author_label_cause: str | None = None
    auto_derivable_crosscheck: str | None = None   # rule-firing auto label, if any
    split_membership: str | None = None            # dev | test | unsampled


# Fields whose presence/non-emptiness the provenance test enforces on every row.
REQUIRED_PROVENANCE_FIELDS = (
    "two_source_papers",      # pair.left.paper_id and pair.right.paper_id, distinct, non-empty
    "source_span_each_side",  # each side has a non-empty evidence_quote or source_block
    "canonical_identity",     # method_id, dataset_id, metric_id all present
    "reported_values",        # both sides have a numeric value
    "noise_decision",         # beyond_noise set with evidence
    "cause_assignment",       # cause in CAUSE_LABELS
    "judge_backbones",        # >= 1 backbone recorded
)


def validate_provenance(row: AttributedRow) -> list[str]:
    """Return a list of provenance violations for one released row (empty == valid)."""
    v: list[str] = []
    p = row.pair
    lp, rp = p.left.paper_id, p.right.paper_id
    if not (lp and rp):
        v.append("two_source_papers: a side is missing paper_id")
    elif lp == rp:
        v.append("two_source_papers: both sides are the same paper (intra-paper, out of scope)")
    if not p.left.has_span():
        v.append("source_span_each_side: left side has no evidence_quote or source_block")
    if not p.right.has_span():
        v.append("source_span_each_side: right side has no evidence_quote or source_block")
    if not (p.method_id and p.dataset_id and p.metric_id):
        v.append("canonical_identity: missing one of method_id/dataset_id/metric_id")
    for side, c in (("left", p.left), ("right", p.right)):
        try:
            float(c.value)
        except (TypeError, ValueError):
            v.append(f"reported_values: {side}.value is not numeric")
    if row.beyond_noise is None or not isinstance(row.noise_evidence, dict) or not row.noise_evidence:
        v.append("noise_decision: beyond_noise/noise_evidence not set")
    if row.cause not in CAUSE_LABELS:
        v.append(f"cause_assignment: cause {row.cause!r} not in CAUSE_LABELS")
    if not row.judge_backbones:
        v.append("judge_backbones: no backbone recorded")
    return v


def row_to_dict(row: AttributedRow) -> dict[str, Any]:
    """Flatten an AttributedRow to a JSON-serializable dict for the released dataset."""
    return asdict(row)

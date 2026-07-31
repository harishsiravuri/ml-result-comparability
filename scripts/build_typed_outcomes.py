"""Item 3: apply the deterministic typed-outcome remap to every judged pair (framework doc
section 5). Deterministic, $0. Emits data/census/typed_outcomes.jsonl.

The frozen census gold is NOT used here: the remap is a function of the extracted protocol
facets, the applied normalizations, and the existing judge cause. No tuning, no thresholds.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.paths import CENSUS  # noqa: E402
from certificates.facets import per_facet, side_protocol, surface_normalizations  # noqa: E402
from certificates.typed_outcomes import EXCLUDED, SIX_OUTCOMES, missing_dimensions_for, typed_outcome  # noqa: E402


def normalizations_for(pair: dict) -> list[dict]:
    """The one deterministic normalization the pipeline applies: percent-vs-fraction rescale."""
    if pair.get("unit_scale_reconciled"):
        return [{"facet": "unit", "op": "percent_to_fraction_rescale", "side": "auto"}]
    return []


def legacy_rows(lp: dict, rp: dict) -> list[dict]:
    """The pre-correction facet view: metric_surface compared as a RAW STRING, so a lexical
    alias ("acc" vs "accuracy") counted as a protocol difference and a one-sided variant
    statement ("overall accuracy" vs "accuracy") counted as a difference rather than as
    unknown. Kept only to report the sensitivity of the correction, never applied."""
    from certificates.facets import INSPECTED_FACETS, relation
    return [{"facet": f, "left": lp.get(f), "right": rp.get(f),
             "relation": relation(lp.get(f), rp.get(f))} for f in INSPECTED_FACETS]


def main():
    cands = [json.loads(l) for l in open(CENSUS / "candidates.jsonl") if l.strip()]
    # existing judge causes (frozen rule-first judge), where the judge ran
    judge = {}
    for src in ("phase5_test_predictions.jsonl", "judge_dev.jsonl"):
        p = CENSUS / src
        if not p.exists():
            continue
        for l in open(p):
            r = json.loads(l)
            if r.get("method") == "judge_frontier" or r.get("backbone") == "frontier":
                judge.setdefault(r["pair_id"], r.get("final_cause", r.get("cause")))

    rows, dist, rule_dist, legacy_dist = [], Counter(), Counter(), Counter()
    n_judged = 0
    for p in cands:
        lp, rp = side_protocol(p["left"]), side_protocol(p["right"])
        pf = per_facet(lp, rp)
        norm = normalizations_for(p) + surface_normalizations(lp, rp)
        miss = missing_dimensions_for(pf)
        cause = judge.get(p["pair_id"])
        if cause:
            n_judged += 1
        t = typed_outcome(pf, norm, miss, cause)
        rows.append({
            "pair_id": p["pair_id"], "typed_outcome": t["typed_outcome"],
            "rule_fired": t["rule"], "requires": t["requires"],
            "judge_cause": cause, "has_judge_trace": bool(cause),
            "per_facet_relations": {r["facet"]: r["relation"] for r in pf},
            "normalizations_applied": [n["op"] for n in norm],
        })
        dist[t["typed_outcome"]] += 1
        rule_dist[t["rule"]] += 1
        lr = legacy_rows(lp, rp)
        legacy_dist[typed_outcome(lr, normalizations_for(p), missing_dimensions_for(lr),
                                  cause)["typed_outcome"]] += 1

    with open(CENSUS / "typed_outcomes.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n = len(rows)
    six = {k: dist.get(k, 0) for k in SIX_OUTCOMES}
    summary = {
        "n_pairs": n, "n_with_judge_trace": n_judged,
        "distribution_six_outcomes": six,
        "excluded_not_a_genuine_pair": dist.get(EXCLUDED, 0),
        "fraction_comparable_under_assumptions": round(
            dist.get("comparable_under_assumptions", 0) / n, 4),
        "fraction_directly_comparable": round(dist.get("directly_comparable", 0) / n, 4),
        "shares": {k: round(v / n, 4) for k, v in dist.items()},
        "rules_fired": dict(rule_dist),
        "sensitivity_legacy_raw_surface_relation": {
            "what": "the same remap with metric_surface compared as a RAW STRING, i.e. before "
                    "the alias / one-sided-variant correction. Reported so the effect of the "
                    "correction is auditable and revertible; NOT the reported distribution.",
            "distribution": dict(legacy_dist),
            "note": "the correction moves lexical aliases out of incompatible and moves "
                    "one-sided variant statements out of partially_comparable into unknown",
        },
    }
    (CENSUS / "typed_outcomes_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

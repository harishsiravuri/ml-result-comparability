"""Gate 4 (Step A): intra-annotator test-retest reliability + auto-derivable cross-check
+ achieved stratification. Deterministic, $0. Decides the AUTO-GATE:
proceed to Phase 5 ONLY IF decision kappa >= 0.60 AND cause kappa >= 0.45.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.paths import CENSUS, CHECKPOINTS  # noqa: E402
from judge.rules import rule_label  # noqa: E402

CAUSES_REAL = {"split", "metric_variant", "evaluation_setting",
               "citation_reporting_discrepancy", "genuine_conflict"}
NOT_REAL = {"within_noise", "extraction_artifact"}
DECISION_KAPPA_GATE = 0.60
CAUSE_KAPPA_GATE = 0.45


def _load(path: Path) -> dict[str, str]:
    return {r["pair_id"]: r["label"].strip() for r in csv.DictReader(open(path)) if r["label"].strip()}


def decision(label: str) -> str:
    return "real" if label in CAUSES_REAL else "not_real"


def main() -> None:
    first = _load(CENSUS / "gold_annotation_sheet.csv")
    retest = _load(CENSUS / "gold_annotation_sheet_retest.csv")
    gold_pkt = {json.loads(l)["pair_id"]: json.loads(l)
                for l in open(CENSUS / "gold_sample.jsonl") if l.strip()}
    cands = {json.loads(l)["pair_id"]: json.loads(l)
             for l in open(CENSUS / "candidates.jsonl") if l.strip()}

    # ---- 1. test-retest on the 45 retest pairs ----
    rt_ids = sorted(set(retest) & set(first))
    f_cause = [first[p] for p in rt_ids]
    r_cause = [retest[p] for p in rt_ids]
    f_dec = [decision(first[p]) for p in rt_ids]
    r_dec = [decision(retest[p]) for p in rt_ids]
    k_dec = float(cohen_kappa_score(f_dec, r_dec))
    k_cause = float(cohen_kappa_score(f_cause, r_cause))
    dec_agree = sum(a == b for a, b in zip(f_dec, r_dec)) / len(rt_ids)
    cause_agree = sum(a == b for a, b in zip(f_cause, r_cause)) / len(rt_ids)

    # ---- 2. auto-derivable cross-check on rule-fired pairs ----
    rule_hits = {"split": [0, 0], "metric_variant": [0, 0]}  # [agree, total]
    rows_xcheck = []
    for pid, lab in first.items():
        rl = rule_label(cands[pid])["rule_label"]
        if rl in ("split", "metric_variant"):
            rule_hits[rl][1] += 1
            if lab == rl:
                rule_hits[rl][0] += 1
            rows_xcheck.append((pid, rl, lab))
    xcheck = {k: {"n": v[1], "agreement": round(v[0] / v[1], 3) if v[1] else None}
              for k, v in rule_hits.items()}
    total_fired = sum(v[1] for v in rule_hits.values())
    total_agree = sum(v[0] for v in rule_hits.values())

    # ---- 3. achieved stratification (label x stratum) ----
    by_label = Counter(first.values())
    n_real = sum(1 for l in first.values() if l in CAUSES_REAL)
    strat = defaultdict(Counter)
    for pid, lab in first.items():
        s = gold_pkt[pid]["stratum"]
        strat["cause_proxy"][s["cause_proxy"]] += 1
        strat["identity_grade"][s["identity_grade"]] += 1
        strat["pair_type"][s["pair_type"]] += 1
        strat["gap_bin"][s["gap_bin"]] += 1
        strat["split_membership"][gold_pkt[pid]["split_membership"]] += 1

    gate_pass = (k_dec >= DECISION_KAPPA_GATE) and (k_cause >= CAUSE_KAPPA_GATE)
    report = {
        "n_gold": len(first), "n_retest": len(rt_ids),
        "test_retest": {
            "decision_cohen_kappa": round(k_dec, 4), "decision_raw_agreement": round(dec_agree, 3),
            "cause_cohen_kappa": round(k_cause, 4), "cause_raw_agreement": round(cause_agree, 3),
        },
        "auto_cross_check": {
            "by_rule": xcheck, "overall_n": total_fired,
            "overall_agreement": round(total_agree / total_fired, 3) if total_fired else None,
            "note": "QA only, not a target",
        },
        "achieved_stratification": {k: dict(v) for k, v in strat.items()},
        "label_distribution": dict(by_label),
        "real_disagreement_count": n_real, "real_disagreement_share": round(n_real / len(first), 3),
        "auto_gate": {
            "decision_kappa_gate": DECISION_KAPPA_GATE, "cause_kappa_gate": CAUSE_KAPPA_GATE,
            "decision_pass": k_dec >= DECISION_KAPPA_GATE, "cause_pass": k_cause >= CAUSE_KAPPA_GATE,
            "PROCEED_TO_PHASE5": bool(gate_pass),
        },
    }
    (CENSUS / "gate4_reliability.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Phase 6: package the released dataset (deterministic, $0).

Emits two artifacts under data/release/ :
  comparekg_gold.jsonl       - the 200 HUMAN-VALIDATED attributed pairs (the headline
                               dataset): full provenance + noise decision + frozen-judge
                               prediction (where the judge ran) + the author label + the
                               auto-derivable cross-check flag + split + reliability flags.
  comparekg_candidates.jsonl - the full 3,058 candidate pairs (the resource): cell identity,
                               source papers + spans, reported values, and the deterministic
                               noise decision + evidence. Judge cause is included only where
                               the frozen judge ran (dev beyond-noise + test-gold); otherwise
                               null and flagged. NO field-wide judge pass was run (kill-gate
                               trip -> census-led; field-wide CAUSE shares are not validated).
Provenance test asserts every released row has two distinct papers and a span each side.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.paths import CENSUS, REPO_ROOT  # noqa: E402
from judge.rules import rule_label  # noqa: E402

RELEASE = REPO_ROOT / "data" / "release"


def main():
    RELEASE.mkdir(parents=True, exist_ok=True)
    cands = {json.loads(l)["pair_id"]: json.loads(l) for l in open(CENSUS / "candidates.jsonl") if l.strip()}
    noise = {json.loads(l)["pair_id"]: json.loads(l) for l in open(CENSUS / "noise_decisions.jsonl") if l.strip()}
    gold = {r["pair_id"]: r for r in csv.DictReader(open(CENSUS / "gold_annotation_sheet.csv"))}
    pkt = {json.loads(l)["pair_id"]: json.loads(l) for l in open(CENSUS / "gold_sample.jsonl") if l.strip()}

    # frozen judge predictions (test-gold from Phase 5 + dev beyond-noise from Phase 3)
    judge = {}
    for src in ("phase5_test_predictions.jsonl", "judge_dev.jsonl"):
        p = CENSUS / src
        if not p.exists():
            continue
        for l in open(p):
            r = json.loads(l)
            if r.get("method") == "judge_frontier" or r.get("backbone") == "frontier":
                judge.setdefault(r["pair_id"], {
                    "cause": r.get("final_cause", r.get("cause")),
                    "top_level": r.get("top_level"), "rationale": r.get("rationale", ""),
                    "rule_label": r.get("rule_label"), "confidence": r.get("confidence"),
                    "backbones": ["anthropic/claude-sonnet-4.6 (frontier)", "deepseek/deepseek-v4-pro (open)"]})

    def side_pointer(side):
        """PUBLIC release: a POINTER only. No excerpt text is redistributed; the source
        location (section/table label) and the reported value are kept so a reader can
        consult the cited arXiv paper directly. arXiv version is not recorded in the frozen
        snapshot, so the id resolves to the arXiv abstract page (latest version)."""
        return {
            "arxiv_id": side["paper_id"],
            "arxiv_abs_url": f"https://arxiv.org/abs/{side['paper_id']}",
            "arxiv_version": None,  # not recorded in the frozen PwC snapshot
            "value": side["value"], "unit": side.get("unit"),
            "split": side.get("split"), "is_own_result": side.get("is_own_result"),
            "source_location": side.get("source_block"),  # section/table LOCATION, not excerpt text
            "quote_verified": side.get("quote_verified"),
            "self_consistency": side.get("self_consistency"),
            "critic_verdict": side.get("critic_verdict"),
        }

    def noise_block(pid):
        nd = noise[pid]
        return {"beyond_noise": nd["beyond_noise"], "range_type": nd["range_type"],
                "gap": nd["gap"], "threshold": nd["threshold"],
                "dispersion_source": [nd["sd_source_left"], nd["sd_source_right"]]}

    def row(pid, with_human):
        p = cands[pid]
        rl = rule_label(p)["rule_label"]
        r = {
            "pair_id": pid, "split": p["split"],
            "cell": {"method_id": p["method_id"], "dataset_id": p["dataset_id"], "metric_id": p["metric_id"],
                     "method": p["method_canonical"], "dataset": p["dataset_canonical"],
                     "metric": p["metric_canonical"], "metric_direction": p["metric_direction"]},
            "identity_grade": p["identity_grade"], "pair_type": p["pair_type"],
            "task_family": p["task_family"], "n_protocols_on_dataset_metric": p["n_protocols_on_dataset_metric"],
            "value_gap": p["value_gap"], "rel_gap": p["rel_gap"], "unit_scale_reconciled": p["unit_scale_reconciled"],
            "left": side_pointer(p["left"]), "right": side_pointer(p["right"]),
            "noise_decision": noise_block(pid),
            "judge_frozen": judge.get(pid),  # null where the judge did not run
            "auto_derivable_crosscheck": rl,  # rule label where a rule fired, else null
        }
        # explicit LABEL PROVENANCE on every row
        human_label = gold[pid]["label"].strip() if (with_human and pid in gold) else None
        jd = judge.get(pid)
        model_suggested = jd["cause"] if jd else None
        r["human_label"] = human_label
        r["model_suggested_label"] = model_suggested
        r["model_confidence"] = (jd.get("confidence") if jd else None)
        r["label_source"] = ("human" if human_label is not None else ("model" if model_suggested is not None else None))
        r["human_validated"] = human_label is not None
        if with_human and pid in gold:
            r["author_label"] = human_label  # retained alias
            r["author_confidence_1to5"] = gold[pid].get("confidence_1to5", "")
            r["in_test_retest"] = bool(pkt[pid]["in_test_retest"])
            r["in_second_annotator"] = bool(pkt[pid]["in_second_annotator"])
        return r

    with open(RELEASE / "comparekg_gold.jsonl", "w") as f:
        for pid in sorted(gold):
            f.write(json.dumps(row(pid, True), ensure_ascii=False) + "\n")
    with open(RELEASE / "comparekg_candidates.jsonl", "w") as f:
        for pid in sorted(cands):
            f.write(json.dumps(row(pid, False), ensure_ascii=False) + "\n")

    n_judge_gold = sum(1 for pid in gold if pid in judge)
    n_judge_all = sum(1 for pid in cands if pid in judge)
    print(json.dumps({
        "gold_rows": len(gold), "candidate_rows": len(cands),
        "gold_rows_with_frozen_judge": n_judge_gold,
        "candidate_rows_with_frozen_judge": n_judge_all,
        "note": "judge ran on test-gold + dev beyond-noise only; no field-wide judge pass",
    }, indent=2))


if __name__ == "__main__":
    main()

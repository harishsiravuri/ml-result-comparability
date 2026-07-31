"""Item A: validate the leaderboard PARTITION itself against the 200 human gold labels
(deterministic, $0). Maps each gold pair onto the partition's pair-level decision
(comparable / incomparable / unknown) and compares to the human protocol decision.
Writes data/census/partition_validation.json.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.paths import CENSUS, REPO_ROOT  # noqa: E402
from noise.stats import wilson_interval  # noqa: E402
from structural.semantics import value_close  # noqa: E402

REAL = {"split", "metric_variant", "evaluation_setting",
        "citation_reporting_discrepancy", "genuine_conflict"}
HUMAN_INCOMPARABLE = {"split", "metric_variant", "evaluation_setting"}     # protocol artifact
HUMAN_COMPARABLE = {"genuine_conflict", "citation_reporting_discrepancy"}  # same protocol


def locate(side_paper, side_method, side_value, lb):
    """Return the cluster index (int) if the side is in a comparable cluster, 'unknown' if
    it is in the comparability-unknown bucket, or None if not locatable."""
    best_ci, best_d = None, None
    for ci, c in enumerate(lb["clusters"]):
        for e in c["ranking"]:
            if e.get("arxiv_id") == side_paper and (e.get("method") == side_method
                                                    or value_close(e["value"], side_value)):
                d = abs(e["value"] - side_value)
                if best_d is None or d < best_d:
                    best_ci, best_d = ci, d
    if best_ci is not None:
        return best_ci
    for e in lb.get("comparability_unknown_entries", []):
        if e.get("paper_id") == side_paper and (e.get("method") == side_method
                                                or value_close(e["value"], side_value)):
            return "unknown"
    return None


def main():
    cands = {json.loads(l)["pair_id"]: json.loads(l) for l in open(CENSUS / "candidates.jsonl") if l.strip()}
    gold = {r["pair_id"]: r["label"].strip() for r in csv.DictReader(open(CENSUS / "gold_annotation_sheet.csv"))}
    lbs = {l["leaderboard_id"]: l for l in
           (json.loads(x) for x in open(REPO_ROOT / "data" / "cleaned_leaderboards" / "cleaned_leaderboards.jsonl"))}

    unmapped = 0
    dropped_not_real = 0
    rows = []  # (partition_decision, human_decision) over mapped real-disagreement pairs
    partition_dist = Counter()
    for pid, lab in gold.items():
        p = cands[pid]
        lid = f"{p['dataset_id']}|{p['metric_id']}"
        lb = lbs.get(lid)
        if lb is None:
            unmapped += 1
            continue
        mc = p["method_canonical"]
        a = locate(p["left"]["paper_id"], mc, float(p["left"]["value"]), lb)
        b = locate(p["right"]["paper_id"], mc, float(p["right"]["value"]), lb)
        if a is None or b is None or a == "unknown" or b == "unknown":
            pdec = "unknown"
        elif a == b:
            pdec = "comparable"
        else:
            pdec = "incomparable"
        partition_dist[pdec] += 1
        if lab not in REAL:
            dropped_not_real += 1
            continue
        hdec = "incomparable" if lab in HUMAN_INCOMPARABLE else "comparable"
        rows.append((pdec, hdec))

    # confusion matrix partition x human (real-disagreement pairs)
    conf = defaultdict(lambda: Counter())
    for pdec, hdec in rows:
        conf[pdec][hdec] += 1

    def wil(k, n):
        p, lo, hi = wilson_interval(k, n)
        return {"n": n, "k": k, "value": round(p, 3), "ci95": [round(lo, 3), round(hi, 3)]}

    # precision(same-cluster -> comparable): of pairs partition=comparable, human=comparable
    comp = conf["comparable"]
    incomp = conf["incomparable"]
    unk = conf["unknown"]
    out = {
        "n_gold": len(gold), "unmapped": unmapped,
        "dropped_not_real_disagreement": dropped_not_real,
        "n_real_disagreement_mapped": len(rows),
        "partition_decision_distribution_all_200": dict(partition_dist),
        "confusion_matrix_partition_x_human_(real_only)": {
            "comparable": dict(comp), "incomparable": dict(incomp), "unknown": dict(unk)},
        "precision_same_cluster_implies_comparable": wil(comp["comparable"], sum(comp.values())),
        "precision_cross_cluster_implies_incomparable": wil(incomp["incomparable"], sum(incomp.values())),
        "unknown_rate_over_real_mapped": wil(sum(unk.values()), len(rows)),
        "unknown_human_disposition": dict(unk),
        "adjudicable_note": "adjudicable real-disagreement pairs (partition comparable or "
                            "incomparable) = %d; the rest are comparability-unknown" %
                            (sum(comp.values()) + sum(incomp.values())),
    }
    (CENSUS / "partition_validation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

"""Paper 2.2 trust-KG sensitivity (SEPARATE from the frozen single-shot; labeled, post-hoc).

Paper 2.2 reuses the SAME resolver/identifiers, so it does not change which candidate pairs
are surfaced; it adds a per-fact calibrated trust probability (calibrated_prob) and a
canonicalization match-type. The human-flagged identity/extraction artifacts concentrate in
hash-matched (partial_pwc) identities. We ask: using Paper 2.2's trust + match-type as a
candidate-surfacing FILTER, how many human-flagged artifacts would we avoid surfacing, and
how does that tighten the census? Reads the Paper 2.2 KG read-only.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.paths import CENSUS, PAPER2_2  # noqa: E402

CAUSES_REAL = {"split", "metric_variant", "evaluation_setting",
               "citation_reporting_discrepancy", "genuine_conflict"}


def main():
    kg = pd.read_parquet(PAPER2_2 / "data" / "graph" / "contribution_kg.parquet")
    # index calibrated_prob by (paper_id, method_id, dataset_id, metric_id, round(value,4))
    kg["vkey"] = kg["value"].round(4)
    prob = {}
    for r in kg.itertuples(index=False):
        prob.setdefault((r.paper_id, r.method_id, r.dataset_id, r.metric_id, round(float(r.value), 4)),
                        []).append(float(r.calibrated_prob))

    gold = {row["pair_id"]: row["label"].strip()
            for row in csv.DictReader(open(CENSUS / "gold_annotation_sheet.csv"))}
    cands = {json.loads(l)["pair_id"]: json.loads(l)
             for l in open(CENSUS / "candidates.jsonl") if l.strip()}

    def side_prob(side):
        key = (side["paper_id"], side["method_id"], side["dataset_id"], side["metric_id"],
               round(float(side["value"]), 4))
        v = prob.get(key)
        return max(v) if v else None

    rows = []
    matched = 0
    for pid, lab in gold.items():
        p = cands[pid]
        pl, pr = side_prob(p["left"]), side_prob(p["right"])
        if pl is not None and pr is not None:
            matched += 1
        rows.append({"pair_id": pid, "label": lab, "is_artifact": lab == "extraction_artifact",
                     "is_real": lab in CAUSES_REAL, "identity_grade": p["identity_grade"],
                     "min_prob": (min(pl, pr) if (pl is not None and pr is not None) else None)})

    # 1. does trust discriminate artifacts? medians + simple ROC-AUC (artifact vs not)
    def median(xs):
        xs = sorted(x for x in xs if x is not None)
        return round(xs[len(xs) // 2], 4) if xs else None

    arti = [r["min_prob"] for r in rows if r["is_artifact"]]
    real = [r["min_prob"] for r in rows if r["is_real"]]
    noise = [r["min_prob"] for r in rows if r["label"] == "within_noise"]

    def auc(pos, neg):  # P(pos_score > neg_score); pos=artifact (expect LOWER prob -> use 1-)
        pos = [x for x in pos if x is not None]; neg = [x for x in neg if x is not None]
        if not pos or not neg:
            return None
        wins = sum((a < b) + 0.5 * (a == b) for a in pos for b in neg)
        return round(wins / (len(pos) * len(neg)), 3)  # AUC of "artifact has LOWER trust"

    # 2. filter sensitivity (raw gold conditional rates + retention)
    def filt_stats(keep):
        kept = [r for r in rows if keep(r)]
        n = len(kept)
        if not n:
            return None
        return {"retained_of_200": n,
                "artifact_rate": round(sum(r["is_artifact"] for r in kept) / n, 3),
                "real_rate": round(sum(r["is_real"] for r in kept) / n, 3),
                "real_retained_frac": round(sum(r["is_real"] for r in kept) / sum(r["is_real"] for r in rows), 3),
                "artifact_retained_frac": round(sum(r["is_artifact"] for r in kept) / sum(r["is_artifact"] for r in rows), 3)}

    med_all = median([r["min_prob"] for r in rows])
    out = {
        "note": "POST-HOC sensitivity, separate from the frozen single-shot. Paper 2.2 reuses "
                "the same resolver/IDs; this tests its trust + match-type as a surfacing filter.",
        "n_gold": len(rows), "n_matched_to_kg": matched,
        "trust_discriminates_artifacts": {
            "median_min_calibrated_prob": {"artifact": median(arti), "real": median(real),
                                           "within_noise": median(noise)},
            "auc_artifact_has_lower_trust": auc(arti, real + noise),
            "reading": "AUC ~0.5 means calibrated_prob does NOT separate identity artifacts "
                       "(expected: value-trust != identity-match correctness).",
        },
        "filter_restrict_to_all_pwc_identity": filt_stats(lambda r: r["identity_grade"] == "all_pwc"),
        "filter_min_calibrated_prob_ge_median": filt_stats(lambda r: r["min_prob"] is not None and r["min_prob"] >= med_all),
        "filter_all_pwc_AND_prob_ge_median": filt_stats(
            lambda r: r["identity_grade"] == "all_pwc" and r["min_prob"] is not None and r["min_prob"] >= med_all),
        "median_min_calibrated_prob_overall": med_all,
        "baseline_full_set": {"artifact_rate": round(sum(r["is_artifact"] for r in rows) / len(rows), 3),
                              "real_rate": round(sum(r["is_real"] for r in rows) / len(rows), 3)},
    }
    (CENSUS / "paper2_2_sensitivity.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

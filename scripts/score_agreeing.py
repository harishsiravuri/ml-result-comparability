"""Item 1: score the agreeing-pairs judge against the human labels (provisional).

The labels are Harish's single-annotator pass; more annotators may follow, so this is scored
as provisional and nothing is frozen. A "protocol barrier" is the human (or model) assigning
split / metric_variant / evaluation_setting rather than comparable / citation_copy.

Reported per arm:
  - barrier RECALL on the 35 cross-protocol pairs (human truth: all 35 are split);
  - FALSE-ALARM rate on the same-observed-protocol stratum (30, all human non-barrier) and on
    the protocol-unknown stratum (human non-barriers only);
  - PRECISION of the barrier prediction, stratum-weighted to the population of 1,287 pairs;
  - leaf-exact agreement and Cohen kappa vs the human.
CIs: cell-clustered bootstrap (the chapter's practice), plus Wilson for the single-stratum
rates. Every statistic is clustered on the canonical (method, dataset, metric) cell.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.paths import CENSUS, RUNS  # noqa: E402
from noise.stats import wilson_interval  # noqa: E402

BARRIER = {"split", "metric_variant", "evaluation_setting"}
BOOT, SEED = 4000, 4242
ARMS = ["open", "frontier", "cues_only", "bare"]


def cell_bootstrap(units, stat, seed=SEED):
    """units: list of (cell_id, record). stat: list-of-records -> float|None.
    Resamples CELLS with replacement."""
    by_cell = defaultdict(list)
    for cid, rec in units:
        by_cell[cid].append(rec)
    cells = list(by_cell)
    rng = random.Random(seed)
    pt = stat([r for _, r in units])
    if pt is None or not cells:
        return [pt, None, None]
    draws = []
    for _ in range(BOOT):
        recs = []
        for _ in cells:
            recs.extend(by_cell[cells[rng.randrange(len(cells))]])
        v = stat(recs)
        if v is not None:
            draws.append(v)
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[int(0.975 * len(draws))]
    return [round(pt, 4), round(lo, 4), round(hi, 4)]


def rate(recs, num, den):
    d = sum(1 for r in recs if den(r))
    if d == 0:
        return None
    return sum(1 for r in recs if den(r) and num(r)) / d


def main():
    human = {r["pair_id"]: r["label"].strip()
             for r in csv.DictReader(open(CENSUS / "agreeing_pairs_sheet.csv"))}
    meta = {json.loads(l)["pair_id"]: json.loads(l)
            for l in open(CENSUS / "agreeing_pairs_meta.jsonl") if l.strip()}
    preds = defaultdict(dict)
    for l in open(CENSUS / "agreeing_judge_predictions.jsonl"):
        r = json.loads(l)
        preds[r["arm"]][r["pair_id"]] = r["leaf"]

    def cell(pid):
        m = meta[pid]
        return (m["method_id"], m["dataset_id"], m["metric_id"])

    def stratum(pid):
        return meta[pid]["protocol"]["protocol_class"]

    ids = sorted(human)
    hb = {p: human[p] in BARRIER for p in ids}
    weight = {"cross_protocol": 1.0, "same_observed_protocol": 8.3333,
              "protocol_unknown": 15.9667}

    # population-level human finding: how the agreeing corpus decomposes
    from collections import Counter
    human_dist = dict(Counter(human[p] for p in ids))

    out = {
        "provisional": True,
        "n_pairs": len(ids),
        "human_label_distribution": human_dist,
        "human_finding_agreement_is_not_comparability": {
            "n_labeled_comparable": human_dist.get("comparable", 0),
            "n_citation_copy": human_dist.get("citation_copy", 0),
            "n_protocol_barrier": sum(1 for p in ids if hb[p]),
            "statement": "of 95 agreeing same-cell pairs, %d are labeled comparable; agreement "
                         "is citation copying (%d) or sits across a protocol boundary (%d)"
                         % (human_dist.get("comparable", 0),
                            human_dist.get("citation_copy", 0),
                            sum(1 for p in ids if hb[p])),
        },
        "strata_n": {s: sum(1 for p in ids if stratum(p) == s)
                     for s in weight},
        "human_barriers_by_stratum": {
            s: sum(1 for p in ids if stratum(p) == s and hb[p]) for s in weight},
        "post_stratification_weights": weight,
        "by_arm": {},
    }

    for arm in ARMS:
        pr = preds[arm]
        mb = {p: pr.get(p, "undetermined") in BARRIER for p in ids}

        # barrier recall on cross_protocol (all 35 are human barriers)
        cp = [(cell(p), p) for p in ids if stratum(p) == "cross_protocol"]
        recall = cell_bootstrap(cp, lambda rs: (sum(mb[p] for p in rs) / len(rs)) if rs else None)
        k_cp, n_cp = sum(mb[p] for _, p in cp), len(cp)
        recall_wilson = list(wilson_interval(k_cp, n_cp))

        # false-alarm on same_observed (all human non-barrier)
        so = [(cell(p), p) for p in ids if stratum(p) == "same_observed_protocol"]
        fa_so = cell_bootstrap(so, lambda rs: (sum(mb[p] for p in rs) / len(rs)) if rs else None)
        k_so = sum(mb[p] for _, p in so)

        # false-alarm on protocol_unknown human-non-barriers only
        pu = [(cell(p), p) for p in ids if stratum(p) == "protocol_unknown" and not hb[p]]
        fa_pu = cell_bootstrap(pu, lambda rs: (sum(mb[p] for p in rs) / len(rs)) if rs else None)
        k_pu = sum(mb[p] for _, p in pu)

        # stratum-weighted precision of the barrier prediction
        def wprec(recs):
            num = den = 0.0
            for p in recs:
                if mb[p]:
                    w = weight[stratum(p)]
                    den += w
                    num += w if hb[p] else 0
            return (num / den) if den else None
        prec = cell_bootstrap([(cell(p), p) for p in ids], wprec)

        # leaf agreement + kappa vs human
        exact = sum(1 for p in ids if pr.get(p) == human[p]) / len(ids)
        from sklearn.metrics import cohen_kappa_score
        kappa = float(cohen_kappa_score([human[p] for p in ids],
                                        [pr.get(p, "undetermined") for p in ids]))
        kbin = float(cohen_kappa_score([hb[p] for p in ids], [mb[p] for p in ids]))

        out["by_arm"][arm] = {
            "barrier_recall_cross_protocol": {
                "k": k_cp, "n": n_cp, "point_lo_hi_cellboot": recall,
                "wilson95": [round(recall_wilson[1], 4), round(recall_wilson[2], 4)]},
            "false_alarm_same_observed": {
                "k": k_so, "n": len(so), "point_lo_hi_cellboot": fa_so},
            "false_alarm_protocol_unknown_nonbarrier": {
                "k": k_pu, "n": len(pu), "point_lo_hi_cellboot": fa_pu},
            "barrier_precision_stratum_weighted": {"point_lo_hi_cellboot": prec},
            "leaf_exact_agreement_vs_human": round(exact, 4),
            "leaf_cohen_kappa_vs_human": round(kappa, 4),
            "binary_barrier_cohen_kappa_vs_human": round(kbin, 4),
            "n_errors": sum(1 for p in ids if p not in pr),
        }

    (CENSUS / "agreeing_scores.json").write_text(json.dumps(out, indent=2))
    d = RUNS / "agreeing"
    d.mkdir(parents=True, exist_ok=True)
    (d / "agreeing_scores.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

"""Revision re-analysis (existing data only, $0): cluster-aware CIs, census totals, the
label-quality table, and the noise-model audit. Writes data/census/revision_analysis.json.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.paths import CENSUS  # noqa: E402

CAUSES_REAL = {"split", "metric_variant", "evaluation_setting",
               "citation_reporting_discrepancy", "genuine_conflict"}
PROTO = {"split", "metric_variant", "evaluation_setting"}
BOOT = 4000
SEED = 4242


def cause_proxy(pair):
    from judge.rules import rule_label
    rl = rule_label(pair)["rule_label"]
    return {"split": "split_differs", "metric_variant": "metric_surface_differs"}.get(rl, "neither")


# ---------- item 1 + 2: cluster-aware census ----------
def census_block():
    cands = [json.loads(l) for l in open(CENSUS / "candidates.jsonl") if l.strip()]
    gold = {r["pair_id"]: r["label"].strip() for r in csv.DictReader(open(CENSUS / "gold_annotation_sheet.csv"))}
    cand_by_id = {p["pair_id"]: p for p in cands}
    rels = sorted(p["rel_gap"] for p in cands)
    cuts = (rels[len(rels) // 3], rels[2 * len(rels) // 3])

    def gapbin(rg):
        return "low" if rg <= cuts[0] else ("mid" if rg <= cuts[1] else "high")

    def stratum(p):
        return (cause_proxy(p), p["identity_grade"], p["pair_type"], gapbin(p["rel_gap"]))

    pop_by_stratum = Counter(stratum(p) for p in cands)
    N = len(cands)
    # gold rows with (stratum, cell, dataset, label)
    rows = []
    for pid, lab in gold.items():
        p = cand_by_id[pid]
        rows.append({"stratum": stratum(p), "cell": (p["method_id"], p["dataset_id"], p["metric_id"]),
                     "dataset": p["dataset_id"], "label": lab})

    def estimate(multiset, catfn):
        by_str = defaultdict(list)
        for r in multiset:
            by_str[r["stratum"]].append(r["label"])
        share = 0.0
        for s, popn in pop_by_stratum.items():
            labs = by_str.get(s)
            if not labs:
                continue
            share += (popn / N) * (sum(1 for l in labs if catfn(l)) / len(labs))
        return share

    cats = {
        "real_disagreement": lambda l: l in CAUSES_REAL,
        "within_noise": lambda l: l == "within_noise",
        "extraction_or_identity_artifact": lambda l: l == "extraction_artifact",
        "protocol_artifact": lambda l: l in PROTO,
        "citation_reporting_discrepancy": lambda l: l == "citation_reporting_discrepancy",
        "genuine_conflict": lambda l: l == "genuine_conflict",
    }
    # clusters
    cells = sorted({r["cell"] for r in rows})
    dsets = sorted({r["dataset"] for r in rows})
    rows_by_cell = defaultdict(list)
    rows_by_ds = defaultdict(list)
    rows_by_str = defaultdict(list)
    for r in rows:
        rows_by_cell[r["cell"]].append(r)
        rows_by_ds[r["dataset"]].append(r)
        rows_by_str[r["stratum"]].append(r)

    def ci(catfn, mode):
        rng = random.Random(SEED)
        point = estimate(rows, catfn)
        vals = []
        for _ in range(BOOT):
            if mode == "pair":  # resample within strata (old)
                ms = []
                for s, rs in rows_by_str.items():
                    ms += [rs[rng.randrange(len(rs))] for _ in rs]
            elif mode == "cell":
                ms = []
                for _ in cells:
                    c = cells[rng.randrange(len(cells))]
                    ms += rows_by_cell[c]
            else:  # dataset
                ms = []
                for _ in dsets:
                    d = dsets[rng.randrange(len(dsets))]
                    ms += rows_by_ds[d]
            vals.append(estimate(ms, catfn))
        vals.sort()
        return [round(point, 4), round(vals[int(0.025 * BOOT)], 4), round(vals[int(0.975 * BOOT)], 4)]

    out = {"n_gold": len(rows), "n_gold_cells": len(cells), "n_gold_datasets": len(dsets)}
    for name, fn in cats.items():
        out[name] = {"point_[lo,hi]_pair": ci(fn, "pair"),
                     "cell_cluster_[lo,hi]": ci(fn, "cell"),
                     "dataset_cluster_[lo,hi]": ci(fn, "dataset")}
    # item 2: totals + residual
    covered = sum(popn / N for s, popn in pop_by_stratum.items() if rows_by_str.get(s))
    real = out["real_disagreement"]["point_[lo,hi]_pair"][0]
    wn = out["within_noise"]["point_[lo,hi]_pair"][0]
    ea = out["extraction_or_identity_artifact"]["point_[lo,hi]_pair"][0]
    out["totals"] = {
        "real": real, "within_noise": wn, "extraction_artifact": ea,
        "sum_three": round(real + wn + ea, 4),
        "uncovered_strata_residual": round(1 - covered, 4),
        "covered_population_mass": round(covered, 4),
        "within_real_breakdown": {"protocol_artifact": out["protocol_artifact"]["point_[lo,hi]_pair"][0],
                                  "citation_reporting_discrepancy": out["citation_reporting_discrepancy"]["point_[lo,hi]_pair"][0],
                                  "genuine_conflict": out["genuine_conflict"]["point_[lo,hi]_pair"][0],
                                  "sum": round(out["protocol_artifact"]["point_[lo,hi]_pair"][0]
                                               + out["citation_reporting_discrepancy"]["point_[lo,hi]_pair"][0]
                                               + out["genuine_conflict"]["point_[lo,hi]_pair"][0], 4)},
    }
    return out


# ---------- item 1: leaderboard-clustered winner-change / three-way ----------
def leaderboard_block():
    lbs = [json.loads(l) for l in open(Path("data/cleaned_leaderboards/cleaned_leaderboards.jsonl")) if l.strip()]
    kd = [l for l in lbs if l["metric_direction"] in ("higher", "lower")]
    rng = random.Random(SEED)

    def boot(vals):
        pt = sum(vals) / len(vals)
        bs = []
        for _ in range(BOOT):
            s = [vals[rng.randrange(len(vals))] for _ in vals]
            bs.append(sum(s) / len(s))
        bs.sort()
        return [round(pt, 4), round(bs[int(0.025 * BOOT)], 4), round(bs[int(0.975 * BOOT)], 4)]

    return {
        "n_leaderboards": len(lbs), "n_known_direction": len(kd),
        "winner_change_leaderboard_bootstrap": boot([1.0 if l["winner_changed"] else 0.0 for l in kd]),
        "pair_comparable_leaderboard_bootstrap": boot([l["pair_comparable_fraction"] for l in lbs]),
        "pair_confirmed_incomparable_leaderboard_bootstrap": boot([l["pair_confirmed_incomparable_fraction"] for l in lbs]),
        "pair_unknown_leaderboard_bootstrap": boot([l["pair_unknown_fraction"] for l in lbs]),
    }


# ---------- item 3: label-quality table ----------
def label_quality_block():
    s = json.load(open(CENSUS / "phase5_scores.json"))
    g4 = json.load(open(CENSUS / "gate4_reliability.json"))
    cf = s["cause_judge_frontier"]
    per = cf["per_cause"]
    macro7 = round(sum(v["f1"] for v in per.values()) / len(per), 4)
    dec = s["decision"]["system_noise+judge"]
    return {
        "decision_real_vs_notreal": {"precision": dec["precision"], "recall": dec["recall"],
                                     "f1": dec["f1"], "f1_ci95": dec["f1_ci95"]},
        "binary_protocol_vs_other_F1": cf["binary_protocol_vs_other_F1"],
        "cause_macroF1_over_%d_causes" % len(per): macro7,
        "top_level_4class_macroF1": cf["top_level_4class_macroF1"],
        "per_cause_[support,P,R,F1]": {c: {"support": v["gold_n"], "precision": v["precision"],
                                           "recall": v["recall"], "f1": v["f1"]} for c, v in per.items()},
        "extraction_artifact_recall": s["extraction_artifact_catch_rate"]["recall"],
        "intra_annotator_test_retest_kappa": {"decision": g4["test_retest"]["decision_cohen_kappa"],
                                              "cause": g4["test_retest"]["cause_cohen_kappa"]},
    }


# ---------- item 5: noise-model audit ----------
def noise_block():
    nd = [json.loads(l) for l in open(CENSUS / "noise_decisions.jsonl") if l.strip()]
    beyond = [r for r in nd if r["beyond_noise"]]
    src = Counter()
    for r in beyond:
        src[r["sd_source_left"]] += 1
        src[r["sd_source_right"]] += 1
    rep = src["reported"]
    tot = src["reported"] + src["defaulted"]
    from noise.model import K, SIGMA_DEFAULT_REL, SIGMA_DEFAULT_BOUNDED_PTS

    def rel(a, b):
        return abs(a - b) / max(abs(a), abs(b), 1e-9)
    grid = [0.02, 0.042, 0.0594, 0.10]
    sens = {}
    for t in grid:
        sens[f"{t}"] = sum(1 for r in nd if rel(*r["reconciled_values"]) > t)
    return {
        "n_candidates": len(nd), "n_beyond_noise": len(beyond),
        "dispersion_source_share_over_beyond_noise": {
            "reported_sides": rep, "defaulted_sides": tot - rep,
            "reported_fraction": round(rep / tot, 4)},
        "default_dispersion": {"sigma_default_rel_unbounded": SIGMA_DEFAULT_REL,
                               "sigma_default_bounded_points": SIGMA_DEFAULT_BOUNDED_PTS, "k": K},
        "decision_formula": "beyond_noise iff |v1-v2| > k*sqrt(sigma1^2+sigma2^2); bounded in "
                            "points (all-defaulted threshold 2.0 pts), unbounded relative "
                            "(all-defaulted 0.0594), unknown requires both views",
        "prevalence_vs_threshold_sensitivity_(rel_gap_screen)": sens,
    }


def main():
    out = {
        "item1_2_census_cluster_aware": census_block(),
        "item1_leaderboard_cluster": leaderboard_block(),
        "item3_label_quality": label_quality_block(),
        "item5_noise_audit": noise_block(),
    }
    (CENSUS / "revision_analysis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

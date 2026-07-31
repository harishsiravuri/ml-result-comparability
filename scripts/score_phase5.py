"""Phase 5 scoring: the FROZEN judge + baselines vs the author gold on the 158 test-gold
pairs. Wilson CIs for proportions, bootstrap CIs for F1. Evaluates the preregistered bars
WITHOUT bending them. Writes data/census/phase5_scores.json.

Evaluation protocol (documented for the report):
  DECISION (all test-gold): real vs not-real.
    gold real iff label in the 5 causes.
    SYSTEM (noise+judge) real iff noise.beyond_noise AND judge.final_cause != extraction_artifact
      (undetermined counts as real). Baselines use their own decision field.
  CAUSE (universe = gold NOT within_noise): isolates the judge's cause attribution from
    the noise model. Uses judge.final_cause (the frozen rule-first output).
    top-level 4-class = {protocol_artifact, citation_reporting_discrepancy, genuine_conflict,
      extraction_artifact}; binary = protocol_artifact vs other; sub-types split, metric_variant.
  Headline judge backbone = frontier (Sonnet 4.6); open backbone reported for robustness.
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
from noise.stats import wilson_interval  # noqa: E402

CAUSES_REAL = {"split", "metric_variant", "evaluation_setting",
               "citation_reporting_discrepancy", "genuine_conflict"}
PROTO = {"split", "metric_variant", "evaluation_setting"}
TOP4 = ("protocol_artifact", "citation_reporting_discrepancy", "genuine_conflict", "extraction_artifact")
BOOT_SEED = 4242
N_BOOT = 2000


def decision(label: str) -> str:
    return "real" if label in CAUSES_REAL else "not_real"


def top_level(label: str) -> str:
    if label in PROTO:
        return "protocol_artifact"
    if label in ("citation_reporting_discrepancy", "genuine_conflict", "extraction_artifact"):
        return label
    return "within_noise"  # only for within_noise / undetermined edge


def _f1_pos(pairs, positive):
    tp = sum(1 for g, p in pairs if g == positive and p == positive)
    fp = sum(1 for g, p in pairs if g != positive and p == positive)
    fn = sum(1 for g, p in pairs if g == positive and p != positive)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def _macro_f1(pairs, classes):
    return sum(_f1_pos(pairs, c)[2] for c in classes) / len(classes)


def _bootstrap(pairs, stat_fn, n=N_BOOT, seed=BOOT_SEED):
    if not pairs:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    point = stat_fn(pairs)
    vals = []
    m = len(pairs)
    for _ in range(n):
        sample = [pairs[rng.randrange(m)] for _ in range(m)]
        vals.append(stat_fn(sample))
    vals.sort()
    return (round(point, 4), round(vals[int(0.025 * n)], 4), round(vals[int(0.975 * n)], 4))


def main():
    # gold (test subset)
    gold_all = {r["pair_id"]: r["label"].strip() for r in csv.DictReader(open(CENSUS / "gold_annotation_sheet.csv"))}
    test_ids = [json.loads(l)["pair_id"] for l in open(CENSUS / "gold_sample.jsonl")
                if l.strip() and json.loads(l)["split_membership"] == "test"]
    gold = {pid: gold_all[pid] for pid in test_ids}
    noise = {json.loads(l)["pair_id"]: json.loads(l) for l in open(CENSUS / "noise_decisions.jsonl") if l.strip()}
    preds = [json.loads(l) for l in open(CENSUS / "phase5_test_predictions.jsonl") if l.strip()]
    by_method = defaultdict(dict)
    for r in preds:
        by_method[r["method"]][r["pair_id"]] = r

    judgeF = by_method["judge_frontier"]
    judgeO = by_method["judge_open"]

    def system_decision(pid, judge):
        nd = noise[pid]["beyond_noise"]
        fc = judge[pid]["final_cause"]
        return "real" if (nd and fc != "extraction_artifact") else "not_real"

    out = {"n_test_gold": len(test_ids), "gold_label_dist": dict(Counter(gold.values())),
           "n_boot": N_BOOT}

    # ---------- DECISION (all test-gold) ----------
    gold_dec = {pid: decision(gold[pid]) for pid in test_ids}
    dec_methods = {}
    # system (noise + frozen judge frontier)
    sys_pairs = [(gold_dec[pid], system_decision(pid, judgeF)) for pid in test_ids]
    dec_methods["system_noise+judge"] = sys_pairs
    # baselines
    for m in ("naive_rule", "frontier_only_controlled", "frontier_only_adversarial",
              "frontier_only_xlineage", "nli_mnli"):
        if m not in by_method:
            continue
        mp = by_method[m]
        pairs = []
        for pid in test_ids:
            if pid not in mp:
                continue
            d = mp[pid].get("decision_disagreement")
            pairs.append((gold_dec[pid], "real" if d else "not_real"))
        dec_methods[m] = pairs
    dec_report = {}
    for m, pairs in dec_methods.items():
        prec, rec, f1 = _f1_pos(pairs, "real")
        f1c = _bootstrap(pairs, lambda pp: _f1_pos(pp, "real")[2])
        dec_report[m] = {"n": len(pairs), "precision": round(prec, 4), "recall": round(rec, 4),
                         "f1": f1c[0], "f1_ci95": [f1c[1], f1c[2]]}
    out["decision"] = dec_report

    # ---------- CAUSE (universe = gold not within_noise) ----------
    cause_ids = [pid for pid in test_ids if gold[pid] != "within_noise"]
    out["n_cause_universe"] = len(cause_ids)

    def cause_block(judge, name):
        # top-level 4-class macro-F1
        t4 = [(top_level(gold[pid]), top_level(judge[pid]["final_cause"])) for pid in cause_ids]
        macro4 = _bootstrap(t4, lambda pp: _macro_f1(pp, TOP4))
        # binary protocol vs other
        binp = [("protocol_artifact" if top_level(gold[pid]) == "protocol_artifact" else "other",
                 "protocol_artifact" if top_level(judge[pid]["final_cause"]) == "protocol_artifact" else "other")
                for pid in cause_ids]
        binf = _bootstrap(binp, lambda pp: _f1_pos(pp, "protocol_artifact")[2])
        binpr = _f1_pos(binp, "protocol_artifact")
        # split + metric_variant sub-type macro-F1 (leaf)
        leaf = [(gold[pid], judge[pid]["final_cause"]) for pid in cause_ids]
        sm = _bootstrap(leaf, lambda pp: (_f1_pos(pp, "split")[2] + _f1_pos(pp, "metric_variant")[2]) / 2)
        # per-cause P/R
        per = {}
        for c in ("split", "metric_variant", "evaluation_setting",
                  "citation_reporting_discrepancy", "genuine_conflict", "extraction_artifact"):
            p, r, f = _f1_pos(leaf, c)
            ng = sum(1 for g, _ in leaf if g == c)
            per[c] = {"gold_n": ng, "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}
        return {
            "top_level_4class_macroF1": macro4[0], "top_level_4class_macroF1_ci95": [macro4[1], macro4[2]],
            "binary_protocol_vs_other_F1": binf[0], "binary_protocol_vs_other_F1_ci95": [binf[1], binf[2]],
            "binary_protocol_precision": round(binpr[0], 3), "binary_protocol_recall": round(binpr[1], 3),
            "split_metricvariant_macroF1": sm[0], "split_metricvariant_macroF1_ci95": [sm[1], sm[2]],
            "per_cause": per,
        }

    out["cause_judge_frontier"] = cause_block(judgeF, "frontier")
    out["cause_judge_open"] = cause_block(judgeO, "open")

    # LLM-only (no rule override) diagnostic for the frontier backbone
    judgeF_llm = {pid: {"final_cause": judgeF[pid]["cause_llm"]} for pid in cause_ids}
    out["cause_judge_frontier_LLMonly_diagnostic"] = cause_block(judgeF_llm, "frontier_llm")

    # baselines on cause (top-level + binary)
    base_cause = {}
    for m in ("naive_rule", "frontier_only_controlled", "frontier_only_adversarial", "frontier_only_xlineage"):
        if m not in by_method:
            continue
        mp = by_method[m]
        def topl(pid):
            r = mp.get(pid, {})
            tl = r.get("top_level") or top_level(r.get("cause", "within_noise"))
            return tl if tl in TOP4 else "within_noise"
        t4 = [(top_level(gold[pid]), topl(pid)) for pid in cause_ids if pid in mp]
        binp = [("protocol_artifact" if top_level(gold[pid]) == "protocol_artifact" else "other",
                 "protocol_artifact" if topl(pid) == "protocol_artifact" else "other")
                for pid in cause_ids if pid in mp]
        base_cause[m] = {
            "top_level_4class_macroF1": _bootstrap(t4, lambda pp: _macro_f1(pp, TOP4)),
            "binary_protocol_vs_other_F1": _bootstrap(binp, lambda pp: _f1_pos(pp, "protocol_artifact")[2]),
        }
    out["cause_baselines"] = base_cause

    # ---------- metric_variant RULE precision/recall vs gold ----------
    rule_mv = [(gold[pid], judgeF[pid]["rule_label"]) for pid in cause_ids]
    tp = sum(1 for g, rl in rule_mv if rl == "metric_variant" and g == "metric_variant")
    fp = sum(1 for g, rl in rule_mv if rl == "metric_variant" and g != "metric_variant")
    fn = sum(1 for g, rl in rule_mv if rl != "metric_variant" and g == "metric_variant")
    out["metric_variant_rule_vs_gold"] = {
        "fired": tp + fp, "gold_metric_variant": tp + fn,
        "precision": round(tp / (tp + fp), 3) if (tp + fp) else None,
        "recall": round(tp / (tp + fn), 3) if (tp + fn) else None,
    }
    # split rule too
    rule_sp = [(gold[pid], judgeF[pid]["rule_label"]) for pid in cause_ids]
    tp = sum(1 for g, rl in rule_sp if rl == "split" and g == "split")
    fp = sum(1 for g, rl in rule_sp if rl == "split" and g != "split")
    fn = sum(1 for g, rl in rule_sp if rl != "split" and g == "split")
    out["split_rule_vs_gold"] = {
        "fired": tp + fp, "gold_split": tp + fn,
        "precision": round(tp / (tp + fp), 3) if (tp + fp) else None,
        "recall": round(tp / (tp + fn), 3) if (tp + fn) else None,
    }

    # ---------- extraction_artifact catch rate (judge recall of extraction_artifact) ----------
    ea_gold = [pid for pid in test_ids if gold[pid] == "extraction_artifact"]
    ea_caught = sum(1 for pid in ea_gold if judgeF[pid]["final_cause"] == "extraction_artifact")
    p, lo, hi = wilson_interval(ea_caught, len(ea_gold))
    out["extraction_artifact_catch_rate"] = {"gold_n": len(ea_gold), "caught": ea_caught,
                                             "recall": round(p, 3), "ci95": [round(lo, 3), round(hi, 3)]}

    # ---------- BAR EVALUATION ----------
    dF1 = out["decision"]["system_noise+judge"]["f1"]
    dF1_ci = out["decision"]["system_noise+judge"]["f1_ci95"]
    base_dec_f1 = {m: out["decision"][m]["f1"] for m in out["decision"] if m != "system_noise+judge"}
    beats_all = all(dF1 > v for v in base_dec_f1.values())
    t4F1 = out["cause_judge_frontier"]["top_level_4class_macroF1"]
    t4_ci = out["cause_judge_frontier"]["top_level_4class_macroF1_ci95"]
    smF1 = out["cause_judge_frontier"]["split_metricvariant_macroF1"]
    sm_ci = out["cause_judge_frontier"]["split_metricvariant_macroF1_ci95"]

    def verdict(point, bar, ci):
        if point >= bar:
            return "PASS"
        return "CLOSE_MISS_escalate" if ci[1] >= bar else "CLEAN_MISS_trip"

    # decision two-condition: floor (>=0.70) AND strictly above all three baselines.
    beats_naive = dF1 > base_dec_f1.get("naive_rule", 1)
    beats_nli = dF1 > base_dec_f1.get("nli_mnli", 1)
    fo = {m: v for m, v in base_dec_f1.items() if m.startswith("frontier_only")}
    beats_all_fo = all(dF1 > v for v in fo.values())
    best_fo = max(fo, key=fo.get) if fo else None
    if dF1 >= 0.70 and beats_all:
        dec_verdict = "PASS"
    elif dF1 >= 0.70 and beats_naive and beats_nli and not beats_all_fo:
        # floor met, beats the wrong-tool baselines, but does not strictly beat the
        # strongest bare frontier model -> close call against frontier-only
        dec_verdict = "FLOOR_MET_but_ties_frontier_only_ESCALATE"
    else:
        dec_verdict = verdict(dF1, 0.70, dF1_ci)
    out["bars"] = {
        "decision_F1>=0.70_and_beats_baselines": {
            "f1": dF1, "ci95": dF1_ci, "bar": 0.70, "meets_floor": dF1 >= 0.70,
            "beats_all_baselines": beats_all, "beats_naive": beats_naive, "beats_nli": beats_nli,
            "beats_all_frontier_only": beats_all_fo, "best_baseline": best_fo,
            "best_baseline_f1": base_dec_f1.get(best_fo) if best_fo else None,
            "baseline_f1s": base_dec_f1, "verdict": dec_verdict},
        "top_level_4class_macroF1>=0.55": {
            "value": t4F1, "ci95": t4_ci, "bar": 0.55, "verdict": verdict(t4F1, 0.55, t4_ci)},
        "split_metricvariant_macroF1>=0.60": {
            "value": smF1, "ci95": sm_ci, "bar": 0.60, "verdict": verdict(smF1, 0.60, sm_ci)},
        "binary_protocol_vs_other_F1_headline": {
            "value": out["cause_judge_frontier"]["binary_protocol_vs_other_F1"],
            "ci95": out["cause_judge_frontier"]["binary_protocol_vs_other_F1_ci95"],
            "note": "well-powered headline detector metric; no hard floor pre-committed"},
    }

    (CENSUS / "phase5_scores.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["bars"], indent=2))
    print("\nfull scores -> data/census/phase5_scores.json")


if __name__ == "__main__":
    main()

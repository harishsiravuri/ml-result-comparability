"""Phase 5 FINDING: the human-validated census (deterministic, $0).

Because the gold sample was STRATIFIED (cause_proxy/identity_grade/pair_type/gap_bin with a
floor on rule-fired strata), raw gold label rates are NOT population rates. We POST-STRATIFY:
population_share(c) = sum_strata (pop_size_s / N_pop) * gold_rate_s(c), with a stratified
bootstrap CI (resample gold within each stratum). This yields a detector-INDEPENDENT,
human-validated prevalence of real disagreement and of the not-real categories, plus the
taxonomy shares among real disagreements, every number carrying identity-grade coverage.
Also computes the leaderboard-audit number and the dev-vs-test delta of the judge.
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
BOOT = 2000
SEED = 4242


def cause_proxy(pair):
    from judge.rules import rule_label
    rl = rule_label(pair)["rule_label"]
    return {"split": "split_differs", "metric_variant": "metric_surface_differs"}.get(rl, "neither")


def main():
    cands = [json.loads(l) for l in open(CENSUS / "candidates.jsonl") if l.strip()]
    gold = {r["pair_id"]: r["label"].strip() for r in csv.DictReader(open(CENSUS / "gold_annotation_sheet.csv"))}
    pkt = {json.loads(l)["pair_id"]: json.loads(l) for l in open(CENSUS / "gold_sample.jsonl") if l.strip()}

    # population strata (must match draw_gold_sample.py logic)
    rels = sorted(p["rel_gap"] for p in cands)
    cuts = (rels[len(rels) // 3], rels[2 * len(rels) // 3])

    def gapbin(rg):
        return "low" if rg <= cuts[0] else ("mid" if rg <= cuts[1] else "high")

    def stratum(p):
        return (cause_proxy(p), p["identity_grade"], p["pair_type"], gapbin(p["rel_gap"]))

    pop_by_stratum = Counter(stratum(p) for p in cands)
    N_pop = len(cands)

    cand_by_id = {p["pair_id"]: p for p in cands}
    gold_by_stratum = defaultdict(list)  # stratum -> [labels]
    for pid, lab in gold.items():
        gold_by_stratum[stratum(cand_by_id[pid])].append(lab)

    def poststrat(category_fn, rng=None):
        """Estimate population share of pairs with category_fn(label)==True."""
        share = 0.0
        for s, popn in pop_by_stratum.items():
            labs = gold_by_stratum.get(s, [])
            if not labs:
                continue  # stratum not sampled (rare); contributes its pop weight to no estimate
            if rng is not None:
                labs = [labs[rng.randrange(len(labs))] for _ in labs]
            rate = sum(1 for l in labs if category_fn(l)) / len(labs)
            share += (popn / N_pop) * rate
        return share

    def ci(category_fn):
        rng = random.Random(SEED)
        point = poststrat(category_fn)
        vals = sorted(poststrat(category_fn, rng) for _ in range(BOOT))
        return [round(point, 4), round(vals[int(0.025 * BOOT)], 4), round(vals[int(0.975 * BOOT)], 4)]

    prevalence = {
        "real_disagreement": ci(lambda l: l in CAUSES_REAL),
        "within_noise": ci(lambda l: l == "within_noise"),
        "extraction_or_identity_artifact": ci(lambda l: l == "extraction_artifact"),
    }
    # taxonomy shares among ALL candidates (population), per cause + protocol-vs-other grain
    shares = {c: ci(lambda l, c=c: l == c) for c in sorted(CAUSES_REAL)}
    shares["protocol_artifact(any sub-type)"] = ci(lambda l: l in PROTO)

    # identity-grade coverage: prevalence restricted to all_pwc gold pairs (raw, high-confidence)
    grade_cov = {}
    for grade in ("all_pwc", "partial_pwc", "hash_only"):
        labs = [gold[pid] for pid in gold if pkt[pid]["stratum"]["identity_grade"] == grade]
        if labs:
            grade_cov[grade] = {"n_gold": len(labs),
                                "real_share_raw": round(sum(1 for l in labs if l in CAUSES_REAL) / len(labs), 3),
                                "extraction_share_raw": round(sum(1 for l in labs if l == "extraction_artifact") / len(labs), 3)}

    # leaderboard-audit: among gold pairs on a shared PwC leaderboard (co-membership>0),
    # share the human calls a protocol artifact (incomparable) vs a real genuine conflict.
    lb_pairs = [(pid, gold[pid]) for pid in gold if cand_by_id[pid]["n_protocols_on_dataset_metric"] > 1]
    lb_real = [(pid, l) for pid, l in lb_pairs if l in CAUSES_REAL]
    audit = {
        "n_gold_on_multiprotocol_dataset_metric": len(lb_pairs),
        "n_of_those_real_disagreement": len(lb_real),
        "protocol_artifact_share_of_real": round(
            sum(1 for _, l in lb_real if l in PROTO) / len(lb_real), 3) if lb_real else None,
        "note": "of human-confirmed real disagreements on a dataset+metric hosting >1 PwC "
                "leaderboard (protocol), the share attributed to a protocol artifact "
                "(i.e. actually incomparable). Coverage = these pairs only.",
    }

    out = {
        "method": "post-stratified gold reweighting (detector-independent, human-validated)",
        "n_gold": len(gold), "n_pop_candidates": N_pop, "n_strata_pop": len(pop_by_stratum),
        "population_prevalence": prevalence,
        "population_taxonomy_shares_among_all_candidates": shares,
        "identity_grade_coverage_raw": grade_cov,
        "leaderboard_audit": audit,
        "caveat": "Validated grain is the inconsistency decision + protocol-vs-other + the "
                  "split sub-type. metric_variant and the finer 4-class taxonomy are a "
                  "measured residual (Phase 5 bars). Shares of finer causes are reported "
                  "but not validated.",
    }
    (CENSUS / "phase5_census.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

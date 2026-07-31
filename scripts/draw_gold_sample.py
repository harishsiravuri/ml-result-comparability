"""Phase 4: draw the preregistered 200-pair human-gold sample (deterministic, $0).

Stratified RANDOM sampling (seed 20260618) across the four preregistered dimensions:
pair_type, identity_grade, value-gap tercile, and the deterministic cause proxy
(split_differs / metric_surface_differs / neither). Proportional largest-remainder
allocation with a floor on the rule-fired strata so the core (split, metric_variant)
is estimable. Drawn from ALL candidates (not only beyond-noise) so the inconsistency
DECISION is validated on positives and negatives.

Emits a BLIND annotation packet (data/census/gold_sample.jsonl): cell identity, both
reported values/units/splits/spans, and the same table/caption/setup context the judge
saw. It contains NO model prediction, NO noise decision, and NO Papers-with-Code curated
value, so the author annotates blind. Also emits a blank annotation sheet and a
stratification summary, and flags the test-retest and second-annotator subsets.
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
from judge.context import fetch_context  # noqa: E402
from judge.rules import rule_label  # noqa: E402

SEED = 20260618
N_TARGET = 200
N_TEST_RETEST = 45          # re-labeled after a gap -> intra-annotator reliability
N_SECOND_ANNOTATOR = 50     # double-labelling overlap hook (kappa slotted in later)


def cause_proxy(pair: dict) -> str:
    rl = rule_label(pair)["rule_label"]
    if rl == "split":
        return "split_differs"
    if rl == "metric_variant":
        return "metric_surface_differs"
    return "neither"


def gap_tercile(rel_gap: float, cuts: tuple[float, float]) -> str:
    if rel_gap <= cuts[0]:
        return "low"
    if rel_gap <= cuts[1]:
        return "mid"
    return "high"


def allocate(strata_sizes: dict, target: int, floor_keys: set) -> dict:
    """Proportional largest-remainder allocation with a floor of 1 on floor_keys."""
    total = sum(strata_sizes.values())
    raw = {k: target * n / total for k, n in strata_sizes.items()}
    alloc = {k: min(int(v), strata_sizes[k]) for k, v in raw.items()}
    # floor on rule-fired strata
    for k in strata_sizes:
        if any(fk in k for fk in floor_keys) and strata_sizes[k] > 0 and alloc[k] == 0:
            alloc[k] = 1
    # largest-remainder top-up to hit target (respecting stratum capacity)
    while sum(alloc.values()) < target:
        rema = sorted(((raw[k] - alloc[k], k) for k in strata_sizes
                       if alloc[k] < strata_sizes[k]), reverse=True)
        if not rema:
            break
        alloc[rema[0][1]] += 1
    # trim if over
    while sum(alloc.values()) > target:
        red = sorted(((alloc[k] - raw[k], k) for k in strata_sizes
                      if alloc[k] > 0 and not any(fk in k for fk in floor_keys)), reverse=True)
        if not red:
            red = sorted(((alloc[k] - raw[k], k) for k in strata_sizes if alloc[k] > 0), reverse=True)
        alloc[red[0][1]] -= 1
    return alloc


def main() -> None:
    pairs = [json.loads(l) for l in open(CENSUS / "candidates.jsonl") if l.strip()]
    rng = random.Random(SEED)

    rels = sorted(p["rel_gap"] for p in pairs)
    cuts = (rels[len(rels) // 3], rels[2 * len(rels) // 3])
    for p in pairs:
        p["_proxy"] = cause_proxy(p)
        p["_gapbin"] = gap_tercile(p["rel_gap"], cuts)
        p["_stratum"] = (p["_proxy"], p["identity_grade"], p["pair_type"], p["_gapbin"])

    by_stratum = defaultdict(list)
    for p in pairs:
        by_stratum[p["_stratum"]].append(p)
    sizes = {k: len(v) for k, v in by_stratum.items()}
    alloc = allocate(sizes, N_TARGET, floor_keys={"split_differs", "metric_surface_differs"})

    chosen = []
    for k, n in alloc.items():
        pool = sorted(by_stratum[k], key=lambda p: p["pair_id"])
        chosen.extend(rng.sample(pool, min(n, len(pool))))
    chosen.sort(key=lambda p: p["pair_id"])
    assert len(chosen) == N_TARGET, f"expected {N_TARGET}, got {len(chosen)}"

    # designate test-retest + second-annotator subsets (reproducible)
    ids = [p["pair_id"] for p in chosen]
    retest = set(rng.sample(ids, N_TEST_RETEST))
    second = set(rng.sample(ids, N_SECOND_ANNOTATOR))

    # ---- blind annotation packet ----
    with open(CENSUS / "gold_sample.jsonl", "w") as f:
        for p in chosen:
            ctx_l = fetch_context(p["left"]["paper_id"], p["left"]["value"],
                                  p["left"]["evidence_quote"], p["left"]["source_block"])
            ctx_r = fetch_context(p["right"]["paper_id"], p["right"]["value"],
                                  p["right"]["evidence_quote"], p["right"]["source_block"])
            rec = {
                "pair_id": p["pair_id"], "split_membership": p["split"],
                "stratum": {"cause_proxy": p["_proxy"], "identity_grade": p["identity_grade"],
                            "pair_type": p["pair_type"], "gap_bin": p["_gapbin"]},
                "in_test_retest": p["pair_id"] in retest,
                "in_second_annotator": p["pair_id"] in second,
                "cell": {"method": p["method_canonical"], "dataset": p["dataset_canonical"],
                         "metric": p["metric_canonical"], "metric_direction": p["metric_direction"]},
                "A": {k: p["left"].get(k) for k in
                      ("paper_id", "value", "unit", "split", "is_own_result",
                       "evidence_quote", "source_block")},
                "B": {k: p["right"].get(k) for k in
                      ("paper_id", "value", "unit", "split", "is_own_result",
                       "evidence_quote", "source_block")},
                "context_A": ctx_l, "context_B": ctx_r,
                # NOTE: deliberately NO model prediction, NO noise decision, NO PwC gold value.
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- blank FIRST-PASS annotation sheet ----
    # The first pass must treat all 200 pairs IDENTICALLY: no retest / second-annotator
    # marker is exposed here, otherwise the intra-annotator test-retest kappa is
    # optimistically biased. Membership is tracked ONLY in the data (gold_sample.jsonl
    # in_test_retest / in_second_annotator); the retest sheet is generated LATER from it
    # by scripts/make_retest_sheet.py, after the first pass is complete.
    with open(CENSUS / "gold_annotation_sheet.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "label", "confidence_1to5", "note"])
        for p in chosen:
            w.writerow([p["pair_id"], "", "", ""])

    # ---- stratification summary ----
    summary = {
        "seed": SEED, "n": len(chosen),
        "gap_tercile_cuts_rel": [round(c, 4) for c in cuts],
        "by_cause_proxy": dict(Counter(p["_proxy"] for p in chosen)),
        "by_identity_grade": dict(Counter(p["identity_grade"] for p in chosen)),
        "by_pair_type": dict(Counter(p["pair_type"] for p in chosen)),
        "by_gap_bin": dict(Counter(p["_gapbin"] for p in chosen)),
        "by_split_membership": dict(Counter(p["split"] for p in chosen)),
        "n_test_retest": len(retest), "n_second_annotator": len(second),
        "n_nonempty_strata": len([k for k, v in alloc.items() if v > 0]),
        "label_space": ["split", "metric_variant", "evaluation_setting",
                        "citation_reporting_discrepancy", "genuine_conflict",
                        "extraction_artifact", "within_noise"],
    }
    (CENSUS / "gold_sample_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote gold_sample.jsonl ({len(chosen)} blind packets), "
          f"gold_annotation_sheet.csv, gold_annotation_sheet_retest.csv")


if __name__ == "__main__":
    main()

"""Item 1: draw the agreeing-pairs validation sample and its BLIND packet ($0).

WHAT THIS TESTS. Not "does the judge still work when the values agree" — the population itself
answers a prior question first. Of the 1,287 agreeing pairs, **none** is `both_own`: two papers
independently producing an identical number essentially never happens, so agreement in this
literature is CITATION COPYING, not reproduction. That is a reportable finding in its own
right, and it means the interesting question is the other one:

    is the comparability decision driven by the PROTOCOL rather than by the VALUE?

The sharp case is an agreeing-value pair that carries a visible split difference. A decision
driven by the value calls those two numbers the same result; a decision driven by the protocol
still flags them incomparable. That is what the oversampled cross-protocol stratum is for.

Population: the 1,287 cross-paper pairs that agree exactly (after the same percent-vs-fraction
reconciliation) on a shared canonical cell. These are precisely the pairs the released census
drops, so the two populations partition the same-cell cross-paper universe.

Frozen-gold firewall: every pair whose canonical cell appears in the 200-pair frozen census
gold is removed BEFORE sampling, so nothing here reuses a cell the annotator has already
judged. The frozen gold files are never read for anything except this exclusion.

Sampling frame: one pair per (cell, protocol_class), taken deterministically as the lowest
pair_id, so the annotator never sees the same cell twice under the same protocol relation.
Allocation deliberately OVERSAMPLES cross-protocol agreement (the stratum the item exists to
test); exact inclusion probabilities are recorded per stratum so population estimates can be
post-stratified back, the same way the census reweights.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from census.agreeing import surface_agreeing  # noqa: E402
from common.paths import CENSUS  # noqa: E402
from judge.context import fetch_context  # noqa: E402

SEED = 20260722                     # fresh seed for this extension; not the census seed
# class -> allocation. cross_protocol is the stratum the item turns on, so it is taken
# EXHAUSTIVELY (its whole one-per-cell frame); weights below restore the population estimate.
ALLOC_BY_CLASS = {"cross_protocol": 35, "same_observed_protocol": 30, "protocol_unknown": 30}
# ONE pair per canonical cell in every stratum (strategic ruling 2026-07-22). A two-per-cell
# cap for cross_protocol would have bought 40 pairs instead of 35, but on a base this small
# full independence is worth more than five pairs and it forecloses any pseudo-replication
# objection. The cross_protocol stratum is therefore a census of its frame, not a sample.
CELL_CAP: dict[str, int] = {}
DEFAULT_CELL_CAP = 1
MIN_CROSS_PROTOCOL = 35


def frozen_gold_cells() -> set[tuple[str, str, str]]:
    cands = {json.loads(l)["pair_id"]: json.loads(l)
             for l in open(CENSUS / "candidates.jsonl") if l.strip()}
    cells = set()
    for line in open(CENSUS / "gold_sample.jsonl"):
        if not line.strip():
            continue
        p = cands[json.loads(line)["pair_id"]]
        cells.add((p["method_id"], p["dataset_id"], p["metric_id"]))
    return cells


def allocate_within(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Proportional-by-identity-grade allocation with a floor of 1 on every present grade."""
    by_grade = defaultdict(list)
    for p in pool:
        by_grade[p["identity_grade"]].append(p)
    sizes = {g: len(v) for g, v in by_grade.items()}
    total = sum(sizes.values())
    alloc = {g: min(int(n * s / total), s) for g, s in sizes.items()}
    for g, s in sizes.items():                       # floor: keep every grade estimable
        if s > 0 and alloc[g] == 0:
            alloc[g] = 1
    while sum(alloc.values()) < n:                   # largest-remainder top-up
        rem = sorted(((n * sizes[g] / total - alloc[g], g) for g in sizes
                      if alloc[g] < sizes[g]), reverse=True)
        if not rem:
            break
        alloc[rem[0][1]] += 1
    while sum(alloc.values()) > n:
        red = sorted(((alloc[g] - n * sizes[g] / total, g) for g in sizes if alloc[g] > 1),
                     reverse=True)
        if not red:
            break
        alloc[red[0][1]] -= 1
    out = []
    for g, k in alloc.items():
        out.extend(rng.sample(sorted(by_grade[g], key=lambda p: p["pair_id"]), min(k, sizes[g])))
    return out


def main() -> None:
    rng = random.Random(SEED)
    all_pairs = surface_agreeing()
    gold_cells = frozen_gold_cells()
    eligible = [p for p in all_pairs
                if (p["method_id"], p["dataset_id"], p["metric_id"]) not in gold_cells]

    # sampling frame: at most CELL_CAP pairs per (cell, protocol_class)
    seen, frame = Counter(), []
    for p in sorted(eligible, key=lambda p: p["pair_id"]):
        cls = p["protocol"]["protocol_class"]
        k = (p["method_id"], p["dataset_id"], p["metric_id"], cls)
        if seen[k] >= CELL_CAP.get(cls, DEFAULT_CELL_CAP):
            continue
        seen[k] += 1
        frame.append(p)

    by_class = defaultdict(list)
    for p in frame:
        by_class[p["protocol"]["protocol_class"]].append(p)

    chosen, weights = [], {}
    for cls, n in ALLOC_BY_CLASS.items():
        pool = by_class.get(cls, [])
        take = allocate_within(pool, min(n, len(pool)), rng)
        for p in take:
            p["_stratum"] = f"{cls}|{p['identity_grade']}"
        chosen.extend(take)
        weights[cls] = {"frame_n": len(pool), "sampled_n": len(take),
                        "inclusion_prob": round(len(take) / len(pool), 5) if pool else None,
                        "post_strat_weight": round(len(pool) / len(take), 4) if take else None}
    chosen.sort(key=lambda p: p["pair_id"])

    # ---- BLIND packet: no model prediction, no deterministic protocol_class, no curated value
    with open(CENSUS / "agreeing_pairs_sample.jsonl", "w") as f:
        for p in chosen:
            ctx_l = fetch_context(p["left"]["paper_id"], p["left"]["value"],
                                  p["left"]["evidence_quote"], p["left"]["source_block"])
            ctx_r = fetch_context(p["right"]["paper_id"], p["right"]["value"],
                                  p["right"]["evidence_quote"], p["right"]["source_block"])
            f.write(json.dumps({
                "pair_id": p["pair_id"],
                "cell": {"method": p["method_canonical"], "dataset": p["dataset_canonical"],
                         "metric": p["metric_canonical"],
                         "metric_direction": p["metric_direction"]},
                "A": {k: p["left"].get(k) for k in
                      ("paper_id", "value", "unit", "split", "is_own_result",
                       "evidence_quote", "source_block")},
                "B": {k: p["right"].get(k) for k in
                      ("paper_id", "value", "unit", "split", "is_own_result",
                       "evidence_quote", "source_block")},
                "context_A": ctx_l, "context_B": ctx_r,
                # deliberately absent: protocol_class, facet relations, model prediction,
                # identity grade, PwC curated value.
            }, ensure_ascii=False) + "\n")

    # ---- the un-blinded record, for scoring AFTER the labels arrive
    with open(CENSUS / "agreeing_pairs_meta.jsonl", "w") as f:
        for p in chosen:
            f.write(json.dumps({k: v for k, v in p.items() if k not in ("left", "right")} |
                               {"left_paper": p["left"]["paper_id"],
                                "right_paper": p["right"]["paper_id"]}) + "\n")

    with open(CENSUS / "agreeing_pairs_sheet.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "label", "confidence_1to5", "note"])
        for p in chosen:
            w.writerow([p["pair_id"], "", "", ""])

    summary = {
        "seed": SEED, "n_sampled": len(chosen),
        "population": {
            "n_agreeing_pairs_total": len(all_pairs),
            "n_after_frozen_gold_cell_exclusion": len(eligible),
            "n_excluded_gold_cell_overlap": len(all_pairs) - len(eligible),
            "n_sampling_frame_one_per_cell_and_class": len(frame),
            "by_protocol_class_population": dict(
                Counter(p["protocol"]["protocol_class"] for p in all_pairs)),
            "by_protocol_class_frame": {k: len(v) for k, v in by_class.items()},
            "by_pair_type_population": dict(Counter(p["pair_type"] for p in all_pairs)),
        },
        "sampled_by_protocol_class": dict(
            Counter(p["protocol"]["protocol_class"] for p in chosen)),
        "sampled_by_identity_grade": dict(Counter(p["identity_grade"] for p in chosen)),
        "sampled_by_pair_type": dict(Counter(p["pair_type"] for p in chosen)),
        "sampled_by_frozen_split_membership": dict(Counter(p["split"] for p in chosen)),
        "sampled_by_differing_facets": dict(
            Counter("+".join(p["protocol"]["differing_facets"]) or "none" for p in chosen)),
        "what_this_tests": "whether the comparability decision is driven by the PROTOCOL "
                           "rather than by the VALUE. The population also establishes that "
                           "agreement in this literature is citation copying, not "
                           "reproduction: 0 of the 1,287 agreeing pairs are both_own.",
        "cross_protocol_floor": {
            "required": MIN_CROSS_PROTOCOL,
            "achieved": sum(1 for p in chosen
                            if p["protocol"]["protocol_class"] == "cross_protocol"),
            "met": sum(1 for p in chosen
                       if p["protocol"]["protocol_class"] == "cross_protocol")
            >= MIN_CROSS_PROTOCOL,
            "cell_cap_applied": CELL_CAP,
            "n_distinct_cells_in_cross_protocol_sample": len(
                {(p["method_id"], p["dataset_id"], p["metric_id"]) for p in chosen
                 if p["protocol"]["protocol_class"] == "cross_protocol"}),
            "n_distinct_datasets_in_cross_protocol_sample": len(
                {p["dataset_id"] for p in chosen
                 if p["protocol"]["protocol_class"] == "cross_protocol"}),
        },
        "n_distinct_datasets": len({p["dataset_id"] for p in chosen}),
        "n_distinct_cells": len({(p["method_id"], p["dataset_id"], p["metric_id"])
                                 for p in chosen}),
        "post_stratification": weights,
        "label_space": ["comparable", "split", "metric_variant", "evaluation_setting",
                        "citation_copy", "extraction_artifact", "undetermined"],
        "blinding": "the packet carries no protocol_class, no facet relation, no identity "
                    "grade and no model prediction",
    }
    (CENSUS / "agreeing_pairs_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

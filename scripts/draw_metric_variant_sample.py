"""Item 5: draw the FRESH metric-variant recovery sample and its BLIND packet ($0).

WHY THE POOL IS NOT "surface-differing pairs". A deterministic audit of all 3,058 census
pairs found that a metric-variant difference is essentially never visible in the metric
SURFACE:

    identical surface        2,740
    one-sided variant stated   165   ("overall accuracy" vs "accuracy": unknown, not different)
    lexical alias only         150   ("acc" vs "accuracy": the same surface, spelled differently)
    genuinely differing variant  3

and, decisively, ALL 11 pairs the author labeled `metric_variant` in the frozen gold have
IDENTICAL metric surfaces. The frozen rule fired on 160 pairs and caught none of the 11, which
is the mechanism behind the 0/17 rule-versus-author agreement recorded at Gate 4.

So metric-variant recovery is a CONTEXT-reading task, not a surface-matching task. 9 of those
11 pairs do carry a variant cue (@k, filtered/raw, micro/macro, ...) in the table caption,
table snippet or setup paragraph. The pool is therefore enriched on that DETERMINISTIC context
cue, with a random no-cue control stratum so the cue's own miss rate stays estimable, plus the
two contaminated strata as negative controls.

Firewalls: every frozen-gold pair and every frozen-gold CELL is excluded, so none of the 11 is
reused and no gold cell is re-labeled. The dev/eval split is by dataset_id under a fresh seed;
eval is SEALED, meaning the metric-variant-aware prompt will be designed on dev labels only.
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from certificates.facets import metric_surface_relation, side_protocol  # noqa: E402
from common.paths import CENSUS  # noqa: E402
from judge.context import fetch_context  # noqa: E402

SEED = 20260722                 # fresh; NOT the census seed (13) or the gold seed (20260618)
DEV_FRACTION = 0.40             # by dataset_id, so no dataset straddles dev and eval
CELL_CAP = 2

# Deterministic variant cues, drawn from the frozen judge's variant vocabulary plus the
# surface forms it misses. Applied to the table caption / snippet / setup paragraph only.
CUE = re.compile(r"(micro|macro|weighted|filtered|\braw\b|per-class|per class|top-1|top-5|"
                 r"@\s*\d+|overall accuracy|mean average|instance-level|frame-level|"
                 r"class-averaged|harmonic mean)", re.I)

ALLOC = {
    "A_identical_surface_with_context_cue": 60,   # enriched: where the known positives live
    "B_identical_surface_no_cue": 30,             # random control: the cue's miss rate
    "C_one_sided_variant_statement": 25,          # what the frozen rule wrongly fired on
    "D_lexical_alias_only": 15,                   # negative control: expected ~0 variants
}


def surface_class(p: dict) -> str:
    lp, rp = side_protocol(p["left"]), side_protocol(p["right"])
    a, b = lp["metric_surface"], rp["metric_surface"]
    if not a or not b:
        return "surface_missing"
    if a == b:
        return "identical_surface"
    rel, _ = metric_surface_relation(a, b)
    return {"observed-different": "genuine_variant_signature",
            "missing": "one_sided_variant_statement"}.get(rel, "lexical_alias_only")


def context_cues(p: dict) -> list[str]:
    hits = set()
    for s in ("left", "right"):
        c = fetch_context(p[s]["paper_id"], p[s]["value"], p[s]["evidence_quote"],
                          p[s]["source_block"])
        if not c or not c.get("available"):
            continue
        txt = " ".join(str(c.get(k) or "")
                       for k in ("table_caption", "table_snippet", "setup_paragraph"))
        hits |= {m.lower().strip() for m in CUE.findall(txt)}
    return sorted(hits)


def stratum_of(p: dict, cues: list[str]) -> str | None:
    sc = surface_class(p)
    if sc == "identical_surface":
        return ("A_identical_surface_with_context_cue" if cues
                else "B_identical_surface_no_cue")
    if sc == "one_sided_variant_statement":
        return "C_one_sided_variant_statement"
    if sc == "lexical_alias_only":
        return "D_lexical_alias_only"
    return None                      # surface_missing / genuine (none eligible) -> out of frame


def main() -> None:
    rng = random.Random(SEED)
    cands = [json.loads(l) for l in open(CENSUS / "candidates.jsonl") if l.strip()]
    byid = {p["pair_id"]: p for p in cands}

    gold_rows = list(csv.DictReader(open(CENSUS / "gold_annotation_sheet.csv")))
    gold_ids = {r["pair_id"] for r in gold_rows}
    gold_mv = {r["pair_id"] for r in gold_rows if r["label"] == "metric_variant"}
    gold_cells = {(byid[i]["method_id"], byid[i]["dataset_id"], byid[i]["metric_id"])
                  for i in gold_ids if i in byid}

    eligible = [p for p in cands
                if p["pair_id"] not in gold_ids
                and (p["method_id"], p["dataset_id"], p["metric_id"]) not in gold_cells]
    assert not (gold_mv & {p["pair_id"] for p in eligible}), "a frozen metric_variant pair leaked"

    by_stratum, cue_of = defaultdict(list), {}
    for p in eligible:
        cues = context_cues(p)
        cue_of[p["pair_id"]] = cues
        s = stratum_of(p, cues)
        if s:
            by_stratum[s].append(p)

    # frame: at most CELL_CAP pairs per canonical cell within a stratum
    frame = defaultdict(list)
    for s, pool in by_stratum.items():
        seen = Counter()
        for p in sorted(pool, key=lambda p: p["pair_id"]):
            k = (p["method_id"], p["dataset_id"], p["metric_id"])
            if seen[k] >= CELL_CAP:
                continue
            seen[k] += 1
            frame[s].append(p)

    chosen, weights = [], {}
    for s, n in ALLOC.items():
        pool = frame.get(s, [])
        take = rng.sample(sorted(pool, key=lambda p: p["pair_id"]), min(n, len(pool)))
        for p in take:
            p["_stratum"] = s
            p["_cues"] = cue_of[p["pair_id"]]
        chosen.extend(take)
        weights[s] = {"population_n": len(by_stratum.get(s, [])), "frame_n": len(pool),
                      "sampled_n": len(take),
                      "post_strat_weight": round(len(by_stratum.get(s, [])) / len(take), 4)
                      if take else None}
    chosen.sort(key=lambda p: p["pair_id"])

    # ---- dev / eval split by dataset_id (fresh seed); EVAL IS SEALED.
    # Datasets are assigned whole (no straddling), but the target is a fraction of PAIRS, not
    # of datasets: splitting on the dataset count alone leaves the pair balance to whichever
    # datasets happen to be pair-heavy.
    per_dataset = Counter(p["dataset_id"] for p in chosen)
    datasets = sorted(per_dataset)
    rng_split = random.Random(SEED + 1)
    rng_split.shuffle(datasets)
    target_dev_pairs = DEV_FRACTION * len(chosen)
    dev_datasets, dev_pairs = set(), 0
    for d in datasets:
        if dev_pairs >= target_dev_pairs:
            break
        dev_datasets.add(d)
        dev_pairs += per_dataset[d]
    for p in chosen:
        p["_fold"] = "dev" if p["dataset_id"] in dev_datasets else "eval"

    # ---- BLIND packet: no stratum, no cue list, no prediction, no curated value
    with open(CENSUS / "metric_variant_sample.jsonl", "w") as f:
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
                      ("paper_id", "value", "unit", "split", "metric", "is_own_result",
                       "evidence_quote", "source_block")},
                "B": {k: p["right"].get(k) for k in
                      ("paper_id", "value", "unit", "split", "metric", "is_own_result",
                       "evidence_quote", "source_block")},
                "context_A": ctx_l, "context_B": ctx_r,
            }, ensure_ascii=False) + "\n")

    # ---- un-blinded record for scoring AFTER labels arrive
    with open(CENSUS / "metric_variant_meta.jsonl", "w") as f:
        for p in chosen:
            f.write(json.dumps({
                "pair_id": p["pair_id"], "stratum": p["_stratum"], "fold": p["_fold"],
                "context_variant_cues": p["_cues"], "surface_class": surface_class(p),
                "left_surface": side_protocol(p["left"])["metric_surface"],
                "right_surface": side_protocol(p["right"])["metric_surface"],
                "identity_grade": p["identity_grade"], "dataset_id": p["dataset_id"],
                "metric_id": p["metric_id"], "method_id": p["method_id"],
                "frozen_split_membership": p["split"], "rel_gap": p["rel_gap"],
            }) + "\n")

    with open(CENSUS / "metric_variant_sheet.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "label", "confidence_1to5", "note"])
        for p in chosen:
            w.writerow([p["pair_id"], "", "", ""])

    summary = {
        "seed": SEED, "n_sampled": len(chosen),
        "why_this_pool": "a metric-variant difference is essentially never visible in the "
                         "metric surface: all 11 frozen-gold metric_variant pairs have "
                         "IDENTICAL surfaces, and 9 of the 11 carry a variant cue in the "
                         "table caption / snippet / setup context. The pool is enriched on "
                         "that deterministic context cue, with a no-cue random control and "
                         "two contaminated strata as negative controls.",
        "firewalls": {
            "frozen_gold_pairs_excluded": len(gold_ids),
            "frozen_gold_metric_variant_pairs_excluded": len(gold_mv),
            "frozen_gold_cells_excluded": len(gold_cells),
            "n_eligible_after_exclusions": len(eligible),
            "any_frozen_metric_variant_pair_in_sample": bool(
                gold_mv & {p["pair_id"] for p in chosen}),
        },
        "surface_class_audit_all_3058_census_pairs": dict(
            Counter(surface_class(p) for p in cands)),
        "strata": weights,
        "sampled_by_stratum": dict(Counter(p["_stratum"] for p in chosen)),
        "sampled_by_fold": dict(Counter(p["_fold"] for p in chosen)),
        "sampled_by_identity_grade": dict(Counter(p["identity_grade"] for p in chosen)),
        "sampled_by_metric": dict(Counter(p["metric_canonical"] for p in chosen).most_common(12)),
        "n_distinct_datasets": len(datasets),
        "n_distinct_cells": len({(p["method_id"], p["dataset_id"], p["metric_id"])
                                 for p in chosen}),
        "dev_eval_split": {"unit": "dataset_id (whole datasets, balanced on PAIR count)",
                           "dev_fraction_target": DEV_FRACTION,
                           "n_dev_datasets": len(dev_datasets),
                           "n_eval_datasets": len(datasets) - len(dev_datasets),
                           "seed": SEED + 1,
                           "seal": "the metric-variant-aware prompt will be designed on DEV "
                                   "labels only; eval is read once, at scoring"},
        "label_space": ["metric_variant", "same_metric_variant", "split",
                        "evaluation_setting", "within_noise",
                        "citation_reporting_discrepancy", "genuine_conflict",
                        "extraction_artifact", "undetermined"],
        "blinding": "the packet carries no stratum, no cue list, no surface class and no "
                    "model prediction",
    }
    (CENSUS / "metric_variant_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

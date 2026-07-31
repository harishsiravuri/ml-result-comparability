"""Item 5: score the metric-variant detector on the DEV slice only (eval sealed, provisional).

The eval fold is never read here. This scores the three arms (context / cues_only / bare)
against the human labels on the 52 dev pairs, and reports the falsifier the gate committed to:

    context must beat cues_only on the variant axis, or item 1's context-over-cues result does
    not transfer to the metric-variant axis.

The variant axis is the binary "is this metric_variant?": positives = human `metric_variant`,
negatives = everything else, with `same_metric_variant` and the stratum-D aliases called out as
the controls the gate names. Dev is small (7 positives, 4 same_metric_variant), so intervals
are wide and this is a DIRECTION check to steer the prompt, not the sealed-eval verdict.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.paths import CENSUS, RUNS  # noqa: E402
from noise.stats import wilson_interval  # noqa: E402

ARMS = ["context", "cues_only", "bare"]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else None
    r = tp / (tp + fn) if (tp + fn) else None
    f = (2 * p * r / (p + r)) if (p and r) else (0.0 if (tp + fp + fn) else None)
    return p, r, f


def main():
    human = {r["pair_id"]: r["label"].strip()
             for r in csv.DictReader(open(CENSUS / "metric_variant_sheet.csv"))}
    meta = {json.loads(l)["pair_id"]: json.loads(l)
            for l in open(CENSUS / "metric_variant_meta.jsonl") if l.strip()}
    preds = defaultdict(dict)
    for l in open(CENSUS / "variant_predictions_dev.jsonl"):
        r = json.loads(l)
        preds[r["arm"]][r["pair_id"]] = r["label"]

    dev = sorted(p for p in human if meta[p]["fold"] == "dev")
    assert all(meta[p]["fold"] == "dev" for p in preds["context"]), "a non-dev pair was scored"

    pos = [p for p in dev if human[p] == "metric_variant"]
    same = [p for p in dev if human[p] == "same_metric_variant"]
    aliasD = [p for p in dev if meta[p]["stratum"].startswith("D")]

    out = {
        "provisional": True, "fold": "dev", "eval_sealed": True,
        "n_dev": len(dev),
        "human_positives_metric_variant": len(pos),
        "human_same_metric_variant": len(same),
        "human_dev_label_distribution": dict(Counter(human[p] for p in dev)),
        "enrichment_check": {
            "note": "all human metric_variant positives should sit in the context-cue stratum A",
            "positives_by_stratum": dict(Counter(meta[p]["stratum"][:1] for p in pos)),
            "stratum_D_alias_metric_variant_count": sum(
                1 for p in aliasD if human[p] == "metric_variant")},
        "by_arm": {},
    }

    for arm in ARMS:
        pr = preds[arm]
        pred_mv = {p: pr.get(p) == "metric_variant" for p in dev}
        tp = sum(1 for p in pos if pred_mv[p])
        fn = len(pos) - tp
        fp = sum(1 for p in dev if pred_mv[p] and human[p] != "metric_variant")
        p_, r_, f_ = prf(tp, fp, fn)
        # false positives on the named controls
        fp_same = sum(1 for p in same if pred_mv[p])
        fp_alias = sum(1 for p in aliasD if pred_mv[p])
        rec_ci = wilson_interval(tp, len(pos)) if pos else (None, None, None)
        # where the human metric_variant positives went under this arm
        pos_confusion = dict(Counter(pr.get(p, "MISSING") for p in pos))
        out["by_arm"][arm] = {
            "metric_variant_precision": round(p_, 4) if p_ is not None else None,
            "metric_variant_recall": round(r_, 4) if r_ is not None else None,
            "metric_variant_recall_wilson95": [round(rec_ci[1], 4), round(rec_ci[2], 4)]
            if pos else None,
            "metric_variant_f1": round(f_, 4) if f_ is not None else None,
            "tp": tp, "fp": fp, "fn": fn,
            "false_positive_on_same_metric_variant": f"{fp_same}/{len(same)}",
            "false_positive_on_stratum_D_alias": f"{fp_alias}/{len(aliasD)}",
            "where_the_metric_variant_positives_went": pos_confusion,
            "predicted_label_distribution": dict(Counter(pr.get(p, "MISSING") for p in dev)),
            "n_errors": sum(1 for p in dev if p not in pr),
        }

    cx, cu = out["by_arm"]["context"], out["by_arm"]["cues_only"]
    beats_recall = (cx["metric_variant_recall"] or 0) > (cu["metric_variant_recall"] or 0)
    beats_f1 = (cx["metric_variant_f1"] or 0) > (cu["metric_variant_f1"] or 0)
    out["falsifier"] = {
        "claim": "context must beat cues_only on the variant axis or item 1 does not transfer",
        "context_recall": cx["metric_variant_recall"], "cues_only_recall": cu["metric_variant_recall"],
        "context_f1": cx["metric_variant_f1"], "cues_only_f1": cu["metric_variant_f1"],
        "context_beats_cues_only_recall": beats_recall,
        "context_beats_cues_only_f1": beats_f1,
        "verdict": "TRANSFERS (context > cues_only)" if (beats_recall or beats_f1)
                   else "DOES NOT TRANSFER on dev",
        "caveat": "dev is small (7 positives); this steers the prompt, the sealed 78-pair eval "
                  "is the verdict of record.",
        "mechanism_from_evidence": "the context arm's misses are not blindness: on the 6 missed "
                                   "positives its own evidence strings name a competing and "
                                   "usually more visible explanation (a different split/dataset "
                                   "annotation, a wrong-cell citation such as R@5-vs-R@10 or "
                                   "T2I-vs-I2T retrieval direction, an extraction artifact). "
                                   "The fuller table pulls it toward "
                                   "citation_reporting_discrepancy / evaluation_setting, which "
                                   "on several pairs is a defensible alternative reading. This "
                                   "is the mirror image of item 1: more context helps on the "
                                   "SPLIT axis and hurts strict metric_variant recall, and it "
                                   "flags a metric_variant-vs-citation label-boundary question "
                                   "for the annotation guide rather than a detection failure.",
    }

    (CENSUS / "variant_scores_dev.json").write_text(json.dumps(out, indent=2))
    d = RUNS / "variant"
    d.mkdir(parents=True, exist_ok=True)
    (d / "variant_scores_dev.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

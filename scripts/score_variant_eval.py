"""Item 5 VERDICT: score the metric-variant detector on the sealed EVAL fold (single-shot).

Eval truth is provisional single-annotator, disclosed as such. The prompt was NOT refined on
dev (the dev result was reported straight, without iteration), so this is a genuine single read
of eval with the first-pass prompt.

Reports, per arm (context / cues_only / bare): metric_variant precision, recall, F1, the alias
false-alarm rate (stratum D), and the plain statement of whether context beats cues_only. A
cell-clustered bootstrap CI accompanies recall and F1; a pooled dev+eval line is added for
context, clearly labelled, since each fold alone is small.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.paths import CENSUS, RUNS  # noqa: E402
from noise.stats import wilson_interval  # noqa: E402

ARMS = ["context", "cues_only", "bare"]
BOOT, SEED = 4000, 4242


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else None
    r = tp / (tp + fn) if (tp + fn) else None
    f = (2 * p * r / (p + r)) if (p and r) else (0.0 if (tp + fp + fn) else None)
    return p, r, f


def cell_boot_f1(units, seed=SEED):
    """units: list of (cell_id, is_pos, is_pred). Cluster-bootstrap F1 and recall over cells."""
    by_cell = defaultdict(list)
    for cid, isp, ispd in units:
        by_cell[cid].append((isp, ispd))
    cells = list(by_cell)
    rng = random.Random(seed)

    def stat(recs):
        tp = sum(1 for isp, ispd in recs if isp and ispd)
        fp = sum(1 for isp, ispd in recs if not isp and ispd)
        fn = sum(1 for isp, ispd in recs if isp and not ispd)
        _, r, f = prf(tp, fp, fn)
        return r, f

    fs, rs = [], []
    for _ in range(BOOT):
        recs = []
        for _ in cells:
            recs.extend(by_cell[cells[rng.randrange(len(cells))]])
        r, f = stat(recs)
        if r is not None:
            rs.append(r)
        if f is not None:
            fs.append(f)
    def ci(xs):
        if not xs:
            return [None, None]
        xs = sorted(xs)
        return [round(xs[int(0.025 * len(xs))], 4), round(xs[int(0.975 * len(xs))], 4)]
    return ci(rs), ci(fs)


def score_fold(fold, human, meta, preds):
    ids = sorted(p for p in human if meta[p]["fold"] == fold)
    pos = [p for p in ids if human[p] == "metric_variant"]
    same = [p for p in ids if human[p] == "same_metric_variant"]
    aliasD = [p for p in ids if meta[p]["stratum"].startswith("D")]
    res = {"n": len(ids), "n_positives": len(pos), "n_same_metric_variant": len(same),
           "positives_by_stratum": dict(Counter(meta[p]["stratum"][:1] for p in pos)),
           "stratum_D_positive_count": sum(1 for p in aliasD if human[p] == "metric_variant"),
           "by_arm": {}}
    for arm in ARMS:
        pr = preds[arm]
        pred = {p: pr.get(p) == "metric_variant" for p in ids}
        tp = sum(1 for p in pos if pred[p])
        fn = len(pos) - tp
        fp = sum(1 for p in ids if pred[p] and human[p] != "metric_variant")
        p_, r_, f_ = prf(tp, fp, fn)
        rec_ci, f1_ci = cell_boot_f1([((meta[p]["method_id"], meta[p]["dataset_id"],
                                        meta[p]["metric_id"]),
                                       human[p] == "metric_variant", pred[p]) for p in ids])
        wil = wilson_interval(tp, len(pos)) if pos else (None, None, None)
        res["by_arm"][arm] = {
            "precision": round(p_, 4) if p_ is not None else None,
            "recall": round(r_, 4) if r_ is not None else None,
            "recall_wilson95": [round(wil[1], 4), round(wil[2], 4)] if pos else None,
            "recall_cellboot95": rec_ci,
            "f1": round(f_, 4) if f_ is not None else None,
            "f1_cellboot95": f1_ci,
            "tp": tp, "fp": fp, "fn": fn,
            "false_positive_on_stratum_D_alias": f"{sum(1 for p in aliasD if pred[p])}/{len(aliasD)}",
            "false_positive_on_same_metric_variant": f"{sum(1 for p in same if pred[p])}/{len(same)}",
            "where_positives_went": dict(Counter(pr.get(p, 'MISSING') for p in pos)),
            "n_errors": sum(1 for p in ids if p not in pr),
        }
    return res


def main():
    human = {r["pair_id"]: r["label"].strip()
             for r in csv.DictReader(open(CENSUS / "metric_variant_sheet.csv"))}
    meta = {json.loads(l)["pair_id"]: json.loads(l)
            for l in open(CENSUS / "metric_variant_meta.jsonl") if l.strip()}

    preds_eval = defaultdict(dict)
    for l in open(CENSUS / "variant_predictions_eval.jsonl"):
        r = json.loads(l)
        preds_eval[r["arm"]][r["pair_id"]] = r["label"]
    assert all(meta[p]["fold"] == "eval" for p in preds_eval["context"]), "non-eval pair scored"

    eval_res = score_fold("eval", human, meta, preds_eval)

    # pooled dev+eval, if dev predictions exist (labelled clearly as pooled, not the verdict)
    pooled = None
    dev_path = CENSUS / "variant_predictions_dev.jsonl"
    if dev_path.exists():
        preds_all = defaultdict(dict)
        for src in (dev_path, CENSUS / "variant_predictions_eval.jsonl"):
            for l in open(src):
                r = json.loads(l)
                preds_all[r["arm"]][r["pair_id"]] = r["label"]
        # score over dev+eval combined by temporarily treating fold membership as "any"
        ids = sorted(p for p in human if p in preds_all["context"])
        m2 = {p: dict(meta[p], fold="_pool") for p in ids}
        for p in ids:
            m2[p]["fold"] = "_pool"
        pooled = score_fold("_pool", human, m2, preds_all)

    cx, cu = eval_res["by_arm"]["context"], eval_res["by_arm"]["cues_only"]
    verdict = {
        "eval_truth_is_provisional_single_annotator": True,
        "prompt_refined_on_dev": False,
        "context_recall": cx["recall"], "cues_only_recall": cu["recall"],
        "context_precision": cx["precision"], "cues_only_precision": cu["precision"],
        "context_f1": cx["f1"], "cues_only_f1": cu["f1"],
        "context_beats_cues_only_recall": (cx["recall"] or 0) > (cu["recall"] or 0),
        "context_beats_cues_only_f1": (cx["f1"] or 0) > (cu["f1"] or 0),
        "context_beats_cues_only": ((cx["recall"] or 0) > (cu["recall"] or 0))
                                   or ((cx["f1"] or 0) > (cu["f1"] or 0)),
    }
    verdict["statement"] = (
        "On the sealed eval fold, context %s cues_only on the metric_variant axis "
        "(context R=%s F1=%s vs cues_only R=%s F1=%s). Item 1's context-over-cues result "
        "%s to the fine metric_variant label." % (
            "BEATS" if verdict["context_beats_cues_only"] else "does NOT beat",
            cx["recall"], cx["f1"], cu["recall"], cu["f1"],
            "TRANSFERS" if verdict["context_beats_cues_only"] else "does NOT transfer"))

    out = {"provisional": True, "fold": "eval", "single_shot": True,
           "eval": eval_res, "pooled_dev_plus_eval_not_the_verdict": pooled,
           "verdict": verdict}
    (CENSUS / "variant_scores_eval.json").write_text(json.dumps(out, indent=2))
    d = RUNS / "variant"
    d.mkdir(parents=True, exist_ok=True)
    (d / "variant_scores_eval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

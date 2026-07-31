"""EARLY GATE: do the structural features separate the human-flagged artifacts / invalid
pairs from the valid ones on the 200-pair dev gold? Deterministic, $0. If they do not
separate at all, STOP (do not build the full system).
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.paths import CENSUS, EXTRACTIONS, RUNS  # noqa: E402
from structural.features import FEATURE_NAMES, build_corpus_index, features_vector  # noqa: E402

CAUSES_REAL = {"split", "metric_variant", "evaluation_setting",
               "citation_reporting_discrepancy", "genuine_conflict"}


def main():
    t0 = time.time()
    df = pd.read_parquet(EXTRACTIONS / "tuples.parquet")
    idx = build_corpus_index(df)
    print(f"corpus index built in {time.time()-t0:.1f}s "
          f"(cells={len(idx['cell_stats'])}, tuples-with-violations={len(idx['tup_viol'])})")

    gold = {r["pair_id"]: r["label"].strip() for r in csv.DictReader(open(CENSUS / "gold_annotation_sheet.csv"))}
    cands = {json.loads(l)["pair_id"]: json.loads(l) for l in open(CENSUS / "candidates.jsonl") if l.strip()}

    X, labels, grades = [], [], []
    for pid, lab in gold.items():
        X.append(features_vector(cands[pid], idx))
        labels.append(lab)
        grades.append(cands[pid]["identity_grade"])
    X = np.array(X, float)
    labels = np.array(labels)

    y_artifact = (labels == "extraction_artifact").astype(int)
    y_notreal = np.array([0 if l in CAUSES_REAL else 1 for l in labels])  # within_noise + artifact
    y_invalid = np.array([0 if l == "genuine_conflict" else 1 for l in labels])  # goal framing (imbalanced)

    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    def combined_auc(y):
        if len(set(y)) < 2:
            return None
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        proba = cross_val_predict(pipe, X, y, cv=5, method="predict_proba")[:, 1]
        return round(float(roc_auc_score(y, proba)), 3)

    # univariate AUCs (top separating features) for the artifact target
    uni = {}
    for j, name in enumerate(FEATURE_NAMES):
        try:
            a = roc_auc_score(y_artifact, X[:, j])
            uni[name] = round(float(max(a, 1 - a)), 3)  # direction-agnostic
        except ValueError:
            uni[name] = None
    top = sorted(((v, k) for k, v in uni.items() if v is not None), reverse=True)[:10]

    out = {
        "n": len(labels), "n_features": len(FEATURE_NAMES),
        "class_counts": {"extraction_artifact": int(y_artifact.sum()),
                         "not_real(within+artifact)": int(y_notreal.sum()),
                         "invalid(!=genuine_conflict)": int(y_invalid.sum()),
                         "genuine_conflict": int((labels == "genuine_conflict").sum())},
        "combined_5foldCV_AUC": {
            "extraction_artifact": combined_auc(y_artifact),
            "not_real": combined_auc(y_notreal),
            "invalid_comparison": combined_auc(y_invalid),
        },
        "top10_univariate_AUC_vs_artifact": [{"feature": k, "auc": v} for v, k in top],
    }
    outdir = RUNS / "sprint"; outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "early_gate.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

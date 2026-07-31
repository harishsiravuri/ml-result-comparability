"""Sprint: structure+LLM comparison-validity detector vs the per-pair frontier model, on the
200-pair dev gold (cross-validated). Uses cached LLM outputs only ($0). Headline: the
extraction/identity-artifact CATCH rate (where corpus structure should beat a per-pair model)
and the decision F1.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.paths import CENSUS, EXTRACTIONS, RUNS  # noqa: E402
from structural.features import FEATURE_NAMES, build_corpus_index, features_vector  # noqa: E402

CAUSES_REAL = {"split", "metric_variant", "evaluation_setting",
               "citation_reporting_discrepancy", "genuine_conflict"}
CAUSE_ORDER = ["split", "metric_variant", "evaluation_setting", "citation_reporting_discrepancy",
               "genuine_conflict", "extraction_artifact", "undetermined", "within_noise"]


def prf(y, pred):
    tp = int(((y == 1) & (pred == 1)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 3), round(r, 3), round(f, 3)


def main():
    gold = {r["pair_id"]: r["label"].strip() for r in csv.DictReader(open(CENSUS / "gold_annotation_sheet.csv"))}
    cands = {json.loads(l)["pair_id"]: json.loads(l) for l in open(CENSUS / "candidates.jsonl") if l.strip()}
    noise = {json.loads(l)["pair_id"]: json.loads(l) for l in open(CENSUS / "noise_decisions.jsonl") if l.strip()}
    llm = {json.loads(l)["pair_id"]: json.loads(l) for l in open(CENSUS / "sprint_llm_gold.jsonl") if l.strip()}
    # frontier-only predictions (controlled = same model; adversarial = scale)
    fo = {"controlled": {}, "adversarial": {}}
    for f in ("data/census/baselines_dev.jsonl", "data/census/phase5_test_predictions.jsonl"):
        for l in open(f):
            r = json.loads(l)
            m = r.get("method", "")
            if m == "frontier_only_controlled":
                fo["controlled"][r["pair_id"]] = r
            elif m == "frontier_only_adversarial":
                fo["adversarial"][r["pair_id"]] = r

    idx = build_corpus_index(pd.read_parquet(EXTRACTIONS / "tuples.parquet"))
    ids = sorted(gold)
    Xs = np.array([features_vector(cands[p], idx) for p in ids], float)
    # LLM features: confidence + cause one-hot
    def onehot(c):
        v = [0.0] * len(CAUSE_ORDER)
        if c in CAUSE_ORDER:
            v[CAUSE_ORDER.index(c)] = 1.0
        return v
    Xllm = np.array([[llm[p]["confidence"]] + onehot(llm[p]["cause_llm"]) for p in ids], float)
    X_struct_llm = np.hstack([Xs, Xllm])
    X_llm_only = Xllm

    labels = np.array([gold[p] for p in ids])
    grades = np.array([cands[p]["identity_grade"] for p in ids])
    beyond = np.array([noise[p]["beyond_noise"] for p in ids])

    y_artifact = (labels == "extraction_artifact").astype(int)
    y_notreal = np.array([0 if l in CAUSES_REAL else 1 for l in labels])  # within+artifact

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict, StratifiedKFold

    cv = StratifiedKFold(5, shuffle=True, random_state=13)

    def cv_pred(X, y):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))
        proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        return (proba >= 0.5).astype(int), proba

    # ---- ARTIFACT catch: fixed baselines vs combiner ----
    art = {}
    # frontier-only: predicts extraction_artifact via top_level
    for tag in ("controlled", "adversarial"):
        pred = np.array([1 if fo[tag].get(p, {}).get("top_level") == "extraction_artifact" else 0 for p in ids])
        art[f"frontier_only_{tag}"] = prf(y_artifact, pred)
    # LLM-only judge (cause_llm)
    pred_llm = np.array([1 if llm[p]["cause_llm"] == "extraction_artifact" else 0 for p in ids])
    art["llm_only_judge"] = prf(y_artifact, pred_llm)
    # combiners (CV)
    pa_struct, _ = cv_pred(Xs, y_artifact)
    pa_sl, _ = cv_pred(X_struct_llm, y_artifact)
    pa_lo, _ = cv_pred(X_llm_only, y_artifact)
    art["structural_only_cv"] = prf(y_artifact, pa_struct)
    art["llm_features_only_cv"] = prf(y_artifact, pa_lo)
    art["STRUCTURE+LLM_cv"] = prf(y_artifact, pa_sl)

    # ---- DECISION (real vs not-real) ----
    dec = {}
    # system with LLM-only: real iff beyond_noise AND cause_llm != extraction_artifact
    sys_llm = np.array([0 if (beyond[i] and llm[ids[i]]["cause_llm"] != "extraction_artifact") else 1
                        for i in range(len(ids))])  # 1 = not_real
    dec["system_noise+LLM-only(no structure)"] = prf(y_notreal, sys_llm)
    for tag in ("controlled", "adversarial"):
        pred = np.array([0 if fo[tag].get(p, {}).get("decision_disagreement") else 1 for p in ids])
        dec[f"frontier_only_{tag}"] = prf(y_notreal, pred)
    pd_sl, _ = cv_pred(X_struct_llm, y_notreal)
    pd_lo, _ = cv_pred(X_llm_only, y_notreal)
    dec["STRUCTURE+LLM_cv"] = prf(y_notreal, pd_sl)
    dec["llm_features_only_cv"] = prf(y_notreal, pd_lo)

    # all_pwc subset artifact-catch (cleanest)
    mask = grades == "all_pwc"
    allpwc = {}
    if mask.sum() >= 10 and y_artifact[mask].sum() >= 2:
        allpwc["n"] = int(mask.sum()); allpwc["n_artifact"] = int(y_artifact[mask].sum())
        allpwc["STRUCTURE+LLM_cv"] = prf(y_artifact[mask], pa_sl[mask])
        allpwc["frontier_only_controlled"] = prf(y_artifact[mask],
            np.array([1 if fo["controlled"].get(p, {}).get("top_level") == "extraction_artifact" else 0 for p in ids])[mask])

    out = {
        "n": len(ids), "n_features_structural": len(FEATURE_NAMES),
        "class_counts": {"extraction_artifact": int(y_artifact.sum()), "not_real": int(y_notreal.sum())},
        "ARTIFACT_catch_[P,R,F1]": art,
        "DECISION_real_vs_notreal_[P,R,F1]_(F1 of not-real)": dec,
        "all_pwc_artifact_catch": allpwc,
        "note": "combiners are 5-fold stratified CV (no training on the fold evaluated); "
                "frontier-only + llm-only are fixed (cached).",
    }
    (RUNS / "sprint" / "combiner_dev.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

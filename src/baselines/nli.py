"""Baseline (c): TEXTUAL NLI / contradiction detector over the paired claim sentences.

An off-the-shelf Natural Language Inference (NLI) cross-encoder (MNLI) classifies whether
one paper's result sentence contradicts the other's. It ignores the quantitative and
protocol dimension, so it can only produce a DECISION (contradiction -> disagreement) and
has no cause taxonomy; it mislabels protocol artifacts as contradictions or non-events.
Runs locally (no API). torch/transformers are imported lazily so this module loads even
before they are installed.
"""

from __future__ import annotations

from functools import lru_cache

# Strongest available open NLI model (verified on HF 2026-06-18); id2label is read
# dynamically so the contradiction class is found regardless of label order. Using the
# strongest NLI model gives this baseline its best shot, which is the honest way to show
# textual NLI is the wrong tool for numeric/protocol result-cell disagreement.
_MODEL_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


@lru_cache(maxsize=1)
def _pipe():
    import torch  # noqa: F401
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(_MODEL_ID)
    model.eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    return tok, model, id2label


def _claim(side: dict, cell: dict) -> str:
    u = side.get("unit") or ""
    return (f"On {cell.get('dataset')}, {cell.get('method')} achieves a "
            f"{cell.get('metric')} of {side.get('value')}{u}"
            f"{(' on the ' + side['split'] + ' split') if side.get('split') else ''}.")


def nli_predict(pair: dict, cell: dict | None = None) -> dict:
    import torch
    tok, model, id2label = _pipe()
    cell = cell or {"method": pair["left"].get("method"), "dataset": pair["left"].get("dataset"),
                    "metric": pair["left"].get("metric")}
    prem = _claim(pair["left"], cell)
    hyp = _claim(pair["right"], cell)
    with torch.no_grad():
        enc = tok(prem, hyp, return_tensors="pt", truncation=True, max_length=256)
        probs = torch.softmax(model(**enc).logits, dim=-1)[0].tolist()
    scores = {id2label[i]: probs[i] for i in range(len(probs))}
    contradiction = scores.get("contradiction", 0.0)
    pred = max(scores, key=scores.get)
    return {
        "pair_id": pair["pair_id"], "method": "nli_mnli", "model_id": _MODEL_ID,
        "nli_label": pred, "contradiction_prob": round(contradiction, 4),
        "decision_disagreement": pred == "contradiction",
        "cause": "nli_no_cause",   # NLI has no cause taxonomy (the point of the baseline)
    }

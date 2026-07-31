# PROVENANCE: copied verbatim 2026-06-18 from paper2_contribqa/src/common/metric_direction.py
#   (Paper 2 ContribQA infrastructure).
# The source repository is READ-ONLY; this is the working copy for paper3.2
#   (comparekg, Chapter 3 cross-paper result-cell disagreement census).
# Do not edit the source; edit this copy if behavior must change here.
"""Metric direction inference from the metric NAME alone (generic knowledge,
no gold data). Shared by gold generation (src/goldgen) and the answering
system (src/answer) — kept in common/ so prompt-side code never imports
gold-side modules."""

from __future__ import annotations

import re

_LOWER_TOKENS = {
    "error", "err", "wer", "cer", "per", "ter", "ppl", "fid", "mae", "mse",
    "rmse", "rmsle", "rms", "epe", "fpr", "eer", "dcf", "mindcf", "mcd", "nll",
    "ece", "fvd", "kid", "loss", "l1", "l2", "d1", "abs", "sq", "swd",
    "regret", "flops", "params", "latency", "runtime", "lpips",
}
_LOWER_PHRASES = [
    "error", "perplexity", "distance", "deviation", "wasserstein", "chamfer",
    "false positive", "false negative", "loss", "time (", "rank error",
]
_HIGHER_TOKENS = {
    "accuracy", "acc", "f1", "f-score", "fscore", "auc", "auroc", "auprc",
    "ap", "map", "ap50", "ap75", "bleu", "rouge", "meteor", "cider", "spice",
    "iou", "miou", "dice", "psnr", "ssim", "ndcg", "mrr", "em", "uas", "las",
    "spearman", "pearson", "kendall", "elo", "r2", "precision", "recall",
    "score", "success", "reward", "return", "moverscore", "bertscore",
}
_HIGHER_PHRASES = [
    "accuracy", "f1", "auc", "precision", "recall", "exact match", "top-1",
    "top-5", "top 1", "top 5", "hit@", "hits@", "r@", "recall@", "ndcg@",
    "map@", "success rate", "win rate", "correlation", "mean iou",
    "average precision", "matthews", "rouge", "bleu", "meteor", "cider",
    "psnr", "ssim", "mrr",
]

_TOKEN_SPLIT = re.compile(r"[^a-z0-9@#^.-]+")


def metric_direction(name: str) -> str:
    """'higher' | 'lower' | 'unknown'. Lower-is-better wins on conflict."""
    n = name.strip().lower()
    tokens = {t for t in _TOKEN_SPLIT.split(n) if t}
    if tokens & _LOWER_TOKENS or any(p in n for p in _LOWER_PHRASES):
        return "lower"
    if tokens & _HIGHER_TOKENS or any(p in n for p in _HIGHER_PHRASES):
        return "higher"
    return "unknown"

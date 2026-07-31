"""Baseline (a): NAIVE RULE. Same canonical identity + differing value -> conflict.

The strawman the taxonomy beats: it has no noise model (so it flags every differing-value
pair as a real disagreement) and no cause taxonomy (so it cannot tell a protocol artifact
from a genuine conflict). We encode its implicit assumption as: decision = disagreement
for every candidate, cause = genuine_conflict. NO LLM, NO gold.
"""

from __future__ import annotations


def naive_predict(pair: dict) -> dict:
    return {
        "pair_id": pair["pair_id"],
        "decision_disagreement": True,      # flags every differing-value pair
        "cause": "genuine_conflict",        # cannot attribute -> assumes a real conflict
        "method": "naive_rule",
    }

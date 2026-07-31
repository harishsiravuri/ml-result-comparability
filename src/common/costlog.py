# PROVENANCE: copied verbatim 2026-06-18 from paper2_contribqa/src/common/costlog.py
#   (Paper 2 ContribQA infrastructure).
# The source repository is READ-ONLY; this is the working copy for paper3.2
#   (comparekg, Chapter 3 cross-paper result-cell disagreement census).
# Do not edit the source; edit this copy if behavior must change here.
"""Append-only cost log + budget guard.

Every real (non-cached) LLM call appends one JSONL record to
experiments/cost_log.jsonl. The budget guard sums cost_usd over the log and
refuses new spend past the cap. Cached replays cost $0 and are not logged
as spend (they get cached=true records only if log_cache_hits is set).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .paths import COST_LOG

HARD_CAP_USD = 200.0


class BudgetExceededError(RuntimeError):
    pass


def cumulative_spend(log_path: Path = COST_LOG) -> float:
    if not log_path.exists():
        return 0.0
    total = 0.0
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                total += float(json.loads(line).get("cost_usd", 0.0))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    return total


def log_call(
    *,
    stage: str,
    model_id: str,
    tokens_in: int,
    tokens_out: int,
    cost_usd: float,
    cache_key: str,
    cached: bool = False,
    log_path: Path = COST_LOG,
) -> None:
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "model_id": model_id,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": cost_usd,
        "cache_key": cache_key,
        "cached": cached,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def check_budget(projected_next_usd: float = 0.0, cap: float = HARD_CAP_USD) -> float:
    """Raise if spend (plus a projected next chunk) would exceed the cap.

    Returns current cumulative spend.
    """
    spend = cumulative_spend()
    if spend + projected_next_usd >= cap:
        raise BudgetExceededError(
            f"Cumulative spend ${spend:.2f} + projected ${projected_next_usd:.2f} "
            f">= cap ${cap:.2f}. STOP per budget protocol."
        )
    return spend

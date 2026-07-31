"""Item 1: run the agreeing-pairs comparability judge over the 100-pair sample.

Cached and resume-safe (sha256 request cache; a re-run costs $0). Temperature 0 throughout.
Four arms, so the labels can be scored against the same comparisons the chapter reports AND so
the structured layer and the table context are separable rather than confounded:

  open       deepseek-v4-pro    + deterministic cues + table/caption/setup context
  frontier   claude-sonnet-4.6  + deterministic cues + table/caption/setup context
  cues_only  claude-opus-4.8    + deterministic cues, NO table context
  bare       claude-opus-4.8    adversarial: NEITHER cues nor context (the chapter's baseline)

Predictions are written to a file the blind packet does not include. Nothing here reads the
frozen census gold, and the annotator never sees any of it.

Usage: python scripts/run_agreeing_judge.py [--arms open,frontier,bare] [--concurrency 8]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import costlog  # noqa: E402
from common.llm import CachedLLM  # noqa: E402
from common.paths import CENSUS, RUNS  # noqa: E402
from judge.agreeing_judge import judge_agreeing  # noqa: E402
from judge.config import model_cfg  # noqa: E402
from judge.context import fetch_context  # noqa: E402

BATCH_BASELINE_USD = 18.676007      # cumulative spend when this extension batch started
BATCH_CAP_USD = 75.0

# Four arms, so the structured layer and the table context are separable rather than
# confounded. `bare` gets NEITHER the deterministic cues nor the table context, which is what
# the chapter's adversarial baseline means; `cues_only` isolates the context contribution.
ARMS = {
    "open": {"cfg": "judge_open", "context": True, "cues": True},
    "frontier": {"cfg": "judge_frontier", "context": True, "cues": True},
    "cues_only": {"cfg": "frontier_only_strongest", "context": False, "cues": True},
    "bare": {"cfg": "frontier_only_strongest", "context": False, "cues": False},
}


def load_sample() -> list[dict]:
    """Rebuild the full pair records for the sampled ids (the packet itself is blind)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "das", Path(__file__).resolve().parent / "draw_agreeing_sample.py")
    das = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(das)
    from census.agreeing import surface_agreeing
    want = {json.loads(l)["pair_id"] for l in open(CENSUS / "agreeing_pairs_sample.jsonl")
            if l.strip()}
    return [p for p in surface_agreeing() if p["pair_id"] in want]


async def run(pairs: list[dict], arms: list[str], concurrency: int) -> list[dict]:
    llm = CachedLLM()
    sem = asyncio.Semaphore(concurrency)
    out: list[dict] = []

    async def one(p, arm):
        cfg = model_cfg(ARMS[arm]["cfg"])
        ctx_l = ctx_r = None
        if ARMS[arm]["context"]:
            ctx_l = fetch_context(p["left"]["paper_id"], p["left"]["value"],
                                  p["left"]["evidence_quote"], p["left"]["source_block"])
            ctx_r = fetch_context(p["right"]["paper_id"], p["right"]["value"],
                                  p["right"]["evidence_quote"], p["right"]["source_block"])
        async with sem:
            o = await judge_agreeing(llm, p, backbone_cfg=cfg, arm=arm,
                                     ctx_left=ctx_l, ctx_right=ctx_r,
                                     include_cues=ARMS[arm]["cues"])
        out.append({
            "pair_id": p["pair_id"], "arm": arm, "model_id": o.backbone,
            "decision": o.decision, "leaf": o.leaf, "sub_type": o.sub_type,
            "confidence": o.confidence, "rationale": o.rationale,
            "cached": o.cached, "cost_usd": o.cost_usd, "error": o.error,
            # the deterministic facet view, for the rule-vs-model comparison after labeling
            "deterministic_protocol_class": p["protocol"]["protocol_class"],
            "deterministic_differing_facets": p["protocol"]["differing_facets"],
        })

    await asyncio.gather(*[one(p, a) for p in pairs for a in arms])
    await llm.aclose()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="open,frontier,cues_only,bare")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    pairs = sorted(load_sample(), key=lambda p: p["pair_id"])
    before = costlog.cumulative_spend()
    print(f"agreeing pairs: {len(pairs)} | arms: {arms} | calls: {len(pairs) * len(arms)} | "
          f"batch spend so far: ${before - BATCH_BASELINE_USD:.4f} / ${BATCH_CAP_USD}")

    results = asyncio.run(run(pairs, arms, args.concurrency))
    after = costlog.cumulative_spend()

    with open(CENSUS / "agreeing_judge_predictions.jsonl", "w") as f:
        for r in sorted(results, key=lambda x: (x["pair_id"], x["arm"])):
            f.write(json.dumps(r) + "\n")

    by_arm = {}
    for a in arms:
        rs = [r for r in results if r["arm"] == a]
        by_arm[a] = {
            "n": len(rs), "n_errors": sum(1 for r in rs if r["error"]),
            "decision_dist": dict(Counter(r["decision"] for r in rs)),
            "leaf_dist": dict(Counter(r["leaf"] for r in rs)),
            "agreement_with_deterministic_cross_protocol": round(sum(
                1 for r in rs
                if (r["decision"] == "protocol_artifact")
                == (r["deterministic_protocol_class"] == "cross_protocol")) / max(len(rs), 1), 4),
        }
    report = {
        "n_pairs": len(pairs), "arms": arms, "n_calls": len(results),
        "spend_this_run_usd": round(after - before, 4),
        "batch_spend_usd": round(after - BATCH_BASELINE_USD, 4),
        "batch_cap_usd": BATCH_CAP_USD,
        "cumulative_spend_all_batches_usd": round(after, 4),
        "by_arm": by_arm,
        "note": "model predictions only; NO human labels exist yet, so nothing here is scored "
                "for correctness. The agreement column compares the model against the "
                "DETERMINISTIC facet class, not against truth.",
    }
    out = RUNS / "agreeing"
    out.mkdir(parents=True, exist_ok=True)
    (out / "agreeing_judge_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

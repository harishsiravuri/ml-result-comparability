"""Phase 5 single-shot: run the FROZEN judge (no retuning) and all baselines ONCE on the
158 test-gold pairs. Cached/resume-safe. Saves predictions for scoring (score_phase5.py).

Methods:
  judge      : frozen rule-first + fuller context, BOTH backbones (open + frontier).
  naive      : code (every differing-value pair -> genuine conflict).
  frontier_only_controlled : judge_frontier model, BARE (no structured layer).
  frontier_only_adversarial: strongest bare model (Opus 4.8).
  frontier_only_xlineage   : strongest non-Anthropic bare model (GPT-5.5).
  nli        : local MNLI contradiction detector.

The judge runs on ALL 158 test-gold pairs (not only beyond-noise), so cause attribution
can be scored on the human-beyond-noise universe. No curated gold value enters any prompt.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.llm import CachedLLM  # noqa: E402
from common.paths import CENSUS, RUNS  # noqa: E402
from common import costlog  # noqa: E402
from judge.config import model_cfg  # noqa: E402
from judge.context import fetch_context  # noqa: E402
from judge.judge import judge_pair  # noqa: E402
from baselines.naive import naive_predict  # noqa: E402
from baselines.frontier_only import frontier_only_predict  # noqa: E402


def _test_gold():
    gold = {json.loads(l)["pair_id"] for l in open(CENSUS / "gold_sample.jsonl")
            if l.strip() and json.loads(l)["split_membership"] == "test"}
    cands = {json.loads(l)["pair_id"]: json.loads(l)
             for l in open(CENSUS / "candidates.jsonl") if l.strip()}
    noise = {json.loads(l)["pair_id"]: json.loads(l)
             for l in open(CENSUS / "noise_decisions.jsonl") if l.strip()}
    items = [{"pair": cands[pid], "noise": noise[pid]} for pid in sorted(gold)]
    return items


async def _run(items, concurrency, skip_nli):
    llm = CachedLLM()
    sem = asyncio.Semaphore(concurrency)
    out = []

    backbones = {"open": model_cfg("judge_open"), "frontier": model_cfg("judge_frontier")}
    fr_cfgs = {"controlled": model_cfg("judge_frontier"),
               "adversarial": model_cfg("frontier_only_strongest"),
               "xlineage": model_cfg("frontier_only_strongest_xlineage")}

    def ctxs(pair):
        cl = fetch_context(pair["left"]["paper_id"], pair["left"]["value"],
                           pair["left"]["evidence_quote"], pair["left"]["source_block"])
        cr = fetch_context(pair["right"]["paper_id"], pair["right"]["value"],
                           pair["right"]["evidence_quote"], pair["right"]["source_block"])
        return cl, cr

    async def judge_one(item, bb, cfg):
        cl, cr = ctxs(item["pair"])
        async with sem:
            o = await judge_pair(llm, item["pair"], item["noise"], backbone_cfg=cfg,
                                 context_arm="context", ctx_left=cl, ctx_right=cr)
        out.append({"pair_id": item["pair"]["pair_id"], "method": f"judge_{bb}",
                    "model_id": o.backbone, "cause_llm": o.cause, "final_cause": o.final_cause,
                    "top_level": o.top_level, "rule_label": o.rule_label,
                    "rule_overridden": o.rule_overridden, "confidence": o.confidence,
                    "error": o.error})

    async def fr_one(item, tag, cfg):
        cl, cr = ctxs(item["pair"])
        async with sem:
            r = await frontier_only_predict(llm, item["pair"], model_cfg=cfg,
                                            ctx_left=cl, ctx_right=cr, tag=tag)
        out.append(r)

    tasks = []
    for it in items:
        for bb, cfg in backbones.items():
            tasks.append(judge_one(it, bb, cfg))
        for tag, cfg in fr_cfgs.items():
            tasks.append(fr_one(it, tag, cfg))
    await asyncio.gather(*tasks)
    await llm.aclose()

    # naive (code) + nli (local)
    for it in items:
        out.append(naive_predict(it["pair"]))
    if not skip_nli:
        from baselines.nli import nli_predict
        for it in items:
            p = it["pair"]
            out.append(nli_predict(p, {"method": p["method_canonical"], "dataset": p["dataset_canonical"],
                                       "metric": p["metric_canonical"]}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--skip-nli", action="store_true")
    args = ap.parse_args()
    items = _test_gold()
    s0 = costlog.cumulative_spend()
    print(f"test-gold pairs: {len(items)} | spend so far ${s0:.4f}")
    preds = asyncio.run(_run(items, args.concurrency, args.skip_nli))
    with open(CENSUS / "phase5_test_predictions.jsonl", "w") as f:
        for r in preds:
            f.write(json.dumps(r) + "\n")
    s1 = costlog.cumulative_spend()
    from collections import Counter
    rep = {"n_test_gold": len(items), "n_predictions": len(preds),
           "by_method": dict(Counter(r["method"] for r in preds)),
           "n_errors": sum(1 for r in preds if r.get("error")),
           "spend_this_run_usd": round(s1 - s0, 4), "cumulative_spend_usd": round(s1, 4)}
    rdir = RUNS / "phase5"; rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "run_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()

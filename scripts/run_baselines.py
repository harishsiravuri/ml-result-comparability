"""Phase 4: run the three baselines (cached, resume-safe). DEV-gold verification by
default; the test-gold run is the Phase 5 single-shot.

Baselines:
  naive        : code-only (flags every differing-value pair as a genuine conflict).
  frontier_only: bare prompt (no structured layer), run controlled (judge_frontier model)
                 and adversarial (strongest bare model). Same fuller context as the judge.
  nli          : local MNLI contradiction detector over paired claim sentences.

Usage:
  python scripts/run_baselines.py --split dev [--limit N] [--skip-nli] [--concurrency 8]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.llm import CachedLLM  # noqa: E402
from common.paths import CENSUS, RUNS  # noqa: E402
from common import costlog  # noqa: E402
from judge.config import model_cfg  # noqa: E402
from judge.context import fetch_context  # noqa: E402
from baselines.naive import naive_predict  # noqa: E402
from baselines.frontier_only import frontier_only_predict  # noqa: E402


def _gold_pairs(split: str | None) -> list[dict]:
    gold = {json.loads(l)["pair_id"]: json.loads(l)
            for l in open(CENSUS / "gold_sample.jsonl") if l.strip()}
    cands = {json.loads(l)["pair_id"]: json.loads(l)
             for l in open(CENSUS / "candidates.jsonl") if l.strip()}
    out = []
    for pid, g in gold.items():
        if split and g["split_membership"] != split:
            continue
        out.append(cands[pid])
    out.sort(key=lambda p: p["pair_id"])
    return out


async def _run_frontier(pairs: list[dict], concurrency: int) -> list[dict]:
    configs = {"controlled": model_cfg("judge_frontier"),
               "adversarial": model_cfg("frontier_only_strongest")}
    llm = CachedLLM()
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = []

    async def one(pair, tag, cfg):
        cl = fetch_context(pair["left"]["paper_id"], pair["left"]["value"],
                           pair["left"]["evidence_quote"], pair["left"]["source_block"])
        cr = fetch_context(pair["right"]["paper_id"], pair["right"]["value"],
                           pair["right"]["evidence_quote"], pair["right"]["source_block"])
        async with sem:
            r = await frontier_only_predict(llm, pair, model_cfg=cfg, ctx_left=cl, ctx_right=cr, tag=tag)
        results.append(r)

    await asyncio.gather(*[one(p, tag, cfg) for p in pairs for tag, cfg in configs.items()])
    await llm.aclose()
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-nli", action="store_true")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    pairs = _gold_pairs(args.split if args.split != "all" else None)
    if args.limit:
        pairs = pairs[: args.limit]
    spend0 = costlog.cumulative_spend()
    print(f"baseline pairs ({args.split}): {len(pairs)} | spend so far ${spend0:.4f}")

    preds: list[dict] = [naive_predict(p) for p in pairs]                       # (a)
    fr = asyncio.run(_run_frontier(pairs, args.concurrency))                    # (b)
    preds.extend(fr)

    nli_preds = []
    if not args.skip_nli:                                                       # (c)
        try:
            from baselines.nli import nli_predict
            for p in pairs:
                cell = {"method": p["method_canonical"], "dataset": p["dataset_canonical"],
                        "metric": p["metric_canonical"]}
                nli_preds.append(nli_predict(p, cell))
        except Exception as e:  # noqa: BLE001
            print(f"[nli] skipped ({type(e).__name__}: {str(e)[:120]})")
    preds.extend(nli_preds)

    out = CENSUS / f"baselines_{args.split}.jsonl"
    with open(out, "w") as f:
        for r in preds:
            f.write(json.dumps(r) + "\n")

    spend1 = costlog.cumulative_spend()
    by_method = Counter(r["method"] for r in preds)
    errs = [r for r in preds if r.get("error")]
    parseable = {m: sum(1 for r in preds if r["method"] == m and r.get("cause") not in (None, "undetermined"))
                 for m in by_method}
    report = {
        "split": args.split, "n_pairs": len(pairs),
        "predictions_by_method": dict(by_method),
        "parseable_nonundetermined_by_method": parseable,
        "n_errors": len(errs),
        "nli_ran": bool(nli_preds),
        "spend_this_run_usd": round(spend1 - spend0, 4),
        "cumulative_spend_usd": round(spend1, 4),
    }
    rdir = RUNS / "phase4_baselines"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / f"baselines_{args.split}_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

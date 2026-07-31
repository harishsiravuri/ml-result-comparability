"""Item 5: run the metric-variant detector over the 130-pair sample (cached, temperature 0).

Three arms mirroring item 1, so the same context-versus-cues contrast is measurable on the
variant axis:

    context     full value-windowed table context (caption + header rows + the value's rows
                + setup). This is the design under test, per the strategic ruling that the
                model must read the TABLE, not a cue list.
    cues_only   deterministic cue strings only, no table text
    bare        neither

FOLD DISCIPLINE. `--fold dev` is the default. The eval fold is SEALED: it is scored once,
after the variant-aware prompt has been settled on dev labels. Running the model on eval
before then is permitted only with --fold all and only to bank a pre-label baseline, since no
labels exist to leak; but any prompt iteration must be driven by dev alone.

Usage:
  python scripts/run_variant_judge.py --fold dev [--arms context,cues_only,bare] [--limit N]
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
from judge.config import model_cfg  # noqa: E402
from judge.context import fetch_context  # noqa: E402
from judge.variant_judge import judge_variant  # noqa: E402

BATCH_BASELINE_USD = 18.676007
BATCH_CAP_USD = 75.0

ARMS = {
    "context": {"cfg": "judge_frontier", "context": True},
    "cues_only": {"cfg": "judge_frontier", "context": False},
    "bare": {"cfg": "judge_frontier", "context": False},
}


def load(fold: str) -> list[dict]:
    cands = {json.loads(l)["pair_id"]: json.loads(l)
             for l in open(CENSUS / "candidates.jsonl") if l.strip()}
    meta = [json.loads(l) for l in open(CENSUS / "metric_variant_meta.jsonl") if l.strip()]
    out = []
    for m in meta:
        if fold != "all" and m["fold"] != fold:
            continue
        p = dict(cands[m["pair_id"]])
        p["_fold"] = m["fold"]
        p["_stratum"] = m["stratum"]
        out.append(p)
    return sorted(out, key=lambda p: p["pair_id"])


async def run(pairs, arms, concurrency):
    llm = CachedLLM()
    sem = asyncio.Semaphore(concurrency)
    out = []

    async def one(p, arm):
        cfg = model_cfg(ARMS[arm]["cfg"])
        ctx_l = ctx_r = None
        if ARMS[arm]["context"]:
            ctx_l = fetch_context(p["left"]["paper_id"], p["left"]["value"],
                                  p["left"]["evidence_quote"], p["left"]["source_block"])
            ctx_r = fetch_context(p["right"]["paper_id"], p["right"]["value"],
                                  p["right"]["evidence_quote"], p["right"]["source_block"])
        async with sem:
            o = await judge_variant(llm, p, backbone_cfg=cfg, arm=arm,
                                    ctx_left=ctx_l, ctx_right=ctx_r)
        out.append({"pair_id": p["pair_id"], "fold": p["_fold"], "stratum": p["_stratum"],
                    "arm": arm, "model_id": o.backbone, "label": o.label,
                    "confidence": o.confidence, "evidence": o.evidence,
                    "cached": o.cached, "cost_usd": o.cost_usd, "error": o.error})

    await asyncio.gather(*[one(p, a) for p in pairs for a in arms])
    await llm.aclose()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="dev", choices=["dev", "eval", "all"])
    ap.add_argument("--arms", default="context,cues_only,bare")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    pairs = load(args.fold)
    if args.limit:
        pairs = pairs[: args.limit]
    before = costlog.cumulative_spend()
    print(f"fold={args.fold} pairs={len(pairs)} arms={arms} calls={len(pairs) * len(arms)} | "
          f"batch spend ${before - BATCH_BASELINE_USD:.4f} / ${BATCH_CAP_USD}")

    results = asyncio.run(run(pairs, arms, args.concurrency))
    after = costlog.cumulative_spend()

    out_path = CENSUS / f"variant_predictions_{args.fold}.jsonl"
    with open(out_path, "w") as f:
        for r in sorted(results, key=lambda x: (x["pair_id"], x["arm"])):
            f.write(json.dumps(r) + "\n")

    by_arm = {}
    for a in arms:
        rs = [r for r in results if r["arm"] == a]
        by_arm[a] = {
            "n": len(rs), "n_errors": sum(1 for r in rs if r["error"]),
            "label_dist": dict(Counter(r["label"] for r in rs)),
            "metric_variant_rate_by_stratum": {
                s: round(sum(1 for r in rs if r["stratum"] == s
                             and r["label"] == "metric_variant")
                         / max(sum(1 for r in rs if r["stratum"] == s), 1), 3)
                for s in sorted({r["stratum"] for r in rs})},
            "cited_evidence_rate": round(sum(
                1 for r in rs if r["evidence"] and r["evidence"].lower().strip()
                not in ("", "none", "n/a")) / max(len(rs), 1), 3),
        }
    report = {
        "fold": args.fold, "n_pairs": len(pairs), "arms": arms, "n_calls": len(results),
        "spend_this_run_usd": round(after - before, 4),
        "batch_spend_usd": round(after - BATCH_BASELINE_USD, 4),
        "batch_cap_usd": BATCH_CAP_USD,
        "by_arm": by_arm,
        "note": "NO human labels exist yet: the per-stratum rates below are model FIRING rates, "
                "not accuracy. Stratum D (lexical alias) is the negative control where the "
                "gate targets a firing rate of 0.",
    }
    d = RUNS / "variant"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"variant_report_{args.fold}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

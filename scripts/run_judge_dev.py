"""Phase 3: run the comparability judge on DEV beyond-noise pairs (cached, resume-safe).

Two backbones (open + frontier) x two context arms (lean spans+cues, fuller +table/caption)
so we can report inter-backbone Cohen kappa and the spans+cues-vs-+context ablation. Tunes
on DEV only; test is untouched. Every call sha256-cached, so re-runs cost $0.

Usage:
  python scripts/run_judge_dev.py [--limit N] [--arms lean,context] [--concurrency 8]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.llm import CachedLLM  # noqa: E402
from common.paths import CENSUS, RUNS  # noqa: E402
from common import costlog  # noqa: E402
from judge.config import model_cfg  # noqa: E402
from judge.context import fetch_context  # noqa: E402
from judge.judge import judge_pair  # noqa: E402


def _load_dev_beyond() -> list[dict]:
    cands = {json.loads(l)["pair_id"]: json.loads(l)
             for l in open(CENSUS / "candidates.jsonl") if l.strip()}
    noise = {json.loads(l)["pair_id"]: json.loads(l)
             for l in open(CENSUS / "noise_decisions.jsonl") if l.strip()}
    out = []
    for pid, nd in noise.items():
        if nd["split"] == "dev" and nd["beyond_noise"]:
            out.append({"pair": cands[pid], "noise": nd})
    out.sort(key=lambda r: r["pair"]["pair_id"])
    return out


async def _run(items: list[dict], arms: list[str], concurrency: int) -> list[dict]:
    backbones = {"open": model_cfg("judge_open"), "frontier": model_cfg("judge_frontier")}
    llm = CachedLLM()
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = []

    async def one(item, bb_name, bb_cfg, arm):
        pair, noise = item["pair"], item["noise"]
        ctx_l = ctx_r = None
        if arm == "context":
            ctx_l = fetch_context(pair["left"]["paper_id"], pair["left"]["value"],
                                  pair["left"]["evidence_quote"], pair["left"]["source_block"])
            ctx_r = fetch_context(pair["right"]["paper_id"], pair["right"]["value"],
                                  pair["right"]["evidence_quote"], pair["right"]["source_block"])
        async with sem:
            o = await judge_pair(llm, pair, noise, backbone_cfg=bb_cfg, context_arm=arm,
                                 ctx_left=ctx_l, ctx_right=ctx_r)
        results.append({
            "pair_id": pair["pair_id"], "split": "dev", "backbone": bb_name,
            "model_id": o.backbone, "arm": arm, "cause": o.cause, "top_level": o.top_level,
            "sub_type": o.sub_type, "confidence": o.confidence, "rule_label": o.rule_label,
            "final_cause": o.final_cause, "rule_overridden": o.rule_overridden,
            "rationale": o.rationale, "cached": o.cached, "cost_usd": o.cost_usd,
            "error": o.error,
        })

    tasks = [one(it, bn, bc, arm) for it in items for bn, bc in backbones.items() for arm in arms]
    await asyncio.gather(*tasks)
    await llm.aclose()
    return results


def _cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    from sklearn.metrics import cohen_kappa_score
    if not labels_a:
        return float("nan")
    return float(cohen_kappa_score(labels_a, labels_b))


def _agreement_report(results: list[dict], arms: list[str]) -> dict:
    by = defaultdict(dict)        # arm -> backbone -> pair_id -> LLM-only cause
    byf = defaultdict(dict)       # arm -> backbone -> pair_id -> rules+LLM final cause
    toplvl = defaultdict(dict)
    rule = {}
    for r in results:
        by[r["arm"]].setdefault(r["backbone"], {})[r["pair_id"]] = r["cause"]
        byf[r["arm"]].setdefault(r["backbone"], {})[r["pair_id"]] = r.get("final_cause", r["cause"])
        toplvl[r["arm"]].setdefault(r["backbone"], {})[r["pair_id"]] = r["top_level"]
        if r["rule_label"]:
            rule[r["pair_id"]] = r["rule_label"]
    rep = {}
    for arm in arms:
        o = by[arm].get("open", {})
        f = by[arm].get("frontier", {})
        common = sorted(set(o) & set(f))
        ka = _cohen_kappa([o[p] for p in common], [f[p] for p in common])
        of = byf[arm].get("open", {})
        ff = byf[arm].get("frontier", {})
        kfinal = _cohen_kappa([of[p] for p in common], [ff[p] for p in common])
        to = toplvl[arm].get("open", {})
        tf = toplvl[arm].get("frontier", {})
        kt = _cohen_kappa([to[p] for p in common], [tf[p] for p in common])
        # agreement with the deterministic rule label, on the rule-fired subset
        rule_acc = {}
        for bb in ("open", "frontier"):
            preds = by[arm].get(bb, {})
            fired = [(preds[p], rule[p]) for p in preds if p in rule]
            hit = sum(1 for pr, rl in fired if pr == rl)
            rule_acc[bb] = {"n_rule_fired": len(fired),
                            "agreement": round(hit / len(fired), 3) if fired else None}
        # abstention + cause distribution per backbone
        dist = {}
        for bb in ("open", "frontier"):
            preds = list(by[arm].get(bb, {}).values())
            c = Counter(preds)
            dist[bb] = {"n": len(preds), "undetermined_rate": round(c["undetermined"] / max(len(preds), 1), 3),
                        "cause_dist": dict(c)}
        rep[arm] = {
            "n_pairs_compared": len(common),
            "cohen_kappa_leaf_cause_LLM_only": round(ka, 4),
            "cohen_kappa_final_rules_plus_LLM": round(kfinal, 4),
            "cohen_kappa_top_level": round(kt, 4),
            "agreement_with_rule_label": rule_acc,
            "per_backbone": dist,
        }
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arms", default="lean,context")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    items = _load_dev_beyond()
    if args.limit:
        items = items[: args.limit]
    spend_before = costlog.cumulative_spend()
    print(f"dev beyond-noise pairs: {len(items)} | arms: {arms} | "
          f"calls: {len(items) * 2 * len(arms)} | spend so far: ${spend_before:.4f}")

    results = asyncio.run(_run(items, arms, args.concurrency))

    out = RUNS / "phase3_judge"
    out.mkdir(parents=True, exist_ok=True)
    with open(CENSUS / "judge_dev.jsonl", "w") as f:
        for r in sorted(results, key=lambda x: (x["pair_id"], x["backbone"], x["arm"])):
            f.write(json.dumps(r) + "\n")

    errors = [r for r in results if r["error"]]
    spend_after = costlog.cumulative_spend()
    report = {
        "n_dev_pairs": len(items), "arms": arms,
        "n_judge_calls": len(results), "n_errors": len(errors),
        "spend_this_run_usd": round(spend_after - spend_before, 4),
        "cumulative_spend_usd": round(spend_after, 4),
        "agreement": _agreement_report(results, arms),
        "error_samples": [e["error"] for e in errors[:5]],
    }
    (out / "judge_dev_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

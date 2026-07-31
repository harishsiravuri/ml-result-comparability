"""Sprint: run the judge (frontier backbone, context arm) on all 200 gold pairs to get the
LLM-ONLY judgment (cause_llm + confidence) for every gold pair. Cached (184/200 already run
in Phase 3/5; ~16 new). Saves data/census/sprint_llm_gold.jsonl.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.llm import CachedLLM  # noqa: E402
from common.paths import CENSUS  # noqa: E402
from common import costlog  # noqa: E402
from judge.config import model_cfg  # noqa: E402
from judge.context import fetch_context  # noqa: E402
from judge.judge import judge_pair  # noqa: E402


async def main():
    gold_ids = [json.loads(l)["pair_id"] for l in open(CENSUS / "gold_sample.jsonl") if l.strip()]
    cands = {json.loads(l)["pair_id"]: json.loads(l) for l in open(CENSUS / "candidates.jsonl") if l.strip()}
    noise = {json.loads(l)["pair_id"]: json.loads(l) for l in open(CENSUS / "noise_decisions.jsonl") if l.strip()}
    cfg = model_cfg("judge_frontier")
    llm = CachedLLM()
    sem = asyncio.Semaphore(10)
    out = []

    async def one(pid):
        p, nd = cands[pid], noise[pid]
        cl = fetch_context(p["left"]["paper_id"], p["left"]["value"], p["left"]["evidence_quote"], p["left"]["source_block"])
        cr = fetch_context(p["right"]["paper_id"], p["right"]["value"], p["right"]["evidence_quote"], p["right"]["source_block"])
        async with sem:
            o = await judge_pair(llm, p, nd, backbone_cfg=cfg, context_arm="context", ctx_left=cl, ctx_right=cr)
        out.append({"pair_id": pid, "cause_llm": o.cause, "top_level": o.top_level,
                    "confidence": o.confidence, "error": o.error})

    s0 = costlog.cumulative_spend()
    await asyncio.gather(*[one(pid) for pid in gold_ids])
    await llm.aclose()
    with open(CENSUS / "sprint_llm_gold.jsonl", "w") as f:
        for r in sorted(out, key=lambda r: r["pair_id"]):
            f.write(json.dumps(r) + "\n")
    print(f"wrote sprint_llm_gold.jsonl ({len(out)} pairs); spend this run ${costlog.cumulative_spend()-s0:.4f}; "
          f"cumulative ${costlog.cumulative_spend():.4f}")


if __name__ == "__main__":
    asyncio.run(main())

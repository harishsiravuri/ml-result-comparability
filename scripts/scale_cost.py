"""Scale-and-cost analysis (deterministic, $0; reads cost_log + existing outputs only).

Concrete cost of doing the field-wide job with a per-pair frontier model versus our
structured pipeline. The frontier-only baseline additionally CANNOT surface the candidates
(that is the structured layer's job), so these projections are a lower bound.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.paths import CENSUS, REPO_ROOT  # noqa: E402

CONCURRENCY = 12
SEC_PER_CALL = 5.0


def main():
    costs = defaultdict(lambda: [0.0, 0])
    for l in open(REPO_ROOT / "experiments" / "cost_log.jsonl"):
        r = json.loads(l)
        costs[r["stage"]][0] += r["cost_usd"]
        costs[r["stage"]][1] += 1

    def per(prefix):
        tot, n = 0.0, 0
        for s, (c, k) in costs.items():
            if s.startswith(prefix):
                tot += c
                n += k
        return (tot / n if n else 0.0, n, tot)

    opus_pc, opus_n, opus_tot = per("frontier_only_adversarial")
    total_spend = round(sum(c for c, _ in costs.values()), 2)

    n_candidates = sum(1 for _ in open(CENSUS / "candidates.jsonl"))
    lbs = [json.loads(l) for l in open(REPO_ROOT / "data" / "cleaned_leaderboards" / "cleaned_leaderboards.jsonl")]
    n_entries = sum(l["n_entries"] for l in lbs)
    field_pairs = sum(math.comb(l["n_entries"], 2) for l in lbs)

    def proj(npairs):
        return {"n_calls": npairs, "cost_usd": round(npairs * opus_pc, 2),
                "wall_hours": round(npairs * SEC_PER_CALL / CONCURRENCY / 3600, 1)}

    out = {
        "frontier_only_per_call_usd": round(opus_pc, 5),
        "our_actual_total_pipeline_spend_usd": total_spend,
        "cross_paper_candidates_surfaced": n_candidates,
        "field_leaderboards": len(lbs), "field_entries": n_entries,
        "field_pairwise_comparisons": field_pairs,
        "projected_frontier_only_judge_all_candidates": proj(n_candidates),
        "projected_frontier_only_field_leaderboard_cleaning": proj(field_pairs),
        "assumptions": {"model": "frontier_only_strongest (Opus 4.8 bare)",
                        "concurrency": CONCURRENCY, "sec_per_call": SEC_PER_CALL},
        "caveat": "The frontier-only baseline cannot SURFACE the cross-paper candidates or the "
                  "leaderboard co-membership; the structured layer does. These projections "
                  "assume the pairs are already surfaced, so they are a lower bound on the cost "
                  "of a per-pair-only approach.",
    }
    (CENSUS / "scale_cost.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

"""AMENDMENT A viability check (deterministic, $0): apply the unified preregistered
noise rule to the candidate set and confirm the beyond-noise population is healthy
(not near the Gate-0 floor of 300). This is a pre-commit feasibility count on the
candidate SET (like Gate 0), NOT the census prevalence finding (which is produced on
test only, after the prereg commit). Also reports the reported-vs-defaulted dispersion
share and the prevalence-versus-threshold sensitivity curve.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.paths import CENSUS, RUNS  # noqa: E402
from census.surface import reconcile  # noqa: E402
from noise.model import SENSITIVITY_REL_GRID, decide  # noqa: E402


def main() -> None:
    pairs = [json.loads(l) for l in open(CENSUS / "candidates.jsonl") if l.strip()]
    beyond = 0
    by_range = Counter()
    by_range_beyond = Counter()
    by_split_beyond = Counter()
    sd_sources = Counter()
    sens = {f"{t}": 0 for t in SENSITIVITY_REL_GRID}
    beyond_pairs_ids = []
    for p in pairs:
        v1, v2 = float(p["left"]["value"]), float(p["right"]["value"])
        x1, x2, _ = reconcile(v1, v2)
        d = decide(x1, x2, p["metric_canonical"] or p["metric_id"],
                   p["left"]["evidence_quote"], p["right"]["evidence_quote"])
        by_range[d.range_type] += 1
        sd_sources[d.sd_source_left] += 1
        sd_sources[d.sd_source_right] += 1
        if d.beyond_noise:
            beyond += 1
            by_range_beyond[d.range_type] += 1
            by_split_beyond[p["split"]] += 1
            if len(beyond_pairs_ids) < 5:
                beyond_pairs_ids.append({
                    "metric": p["metric_canonical"], "range": d.range_type,
                    "gap": round(d.gap, 3), "thr": round(d.threshold, 3),
                    "v": [round(x1, 3), round(x2, 3)],
                })
        # sensitivity: simple relative-gap screen (comparable across thresholds)
        rg = abs(x1 - x2) / max(abs(x1), abs(x2), 1e-9)
        for t in SENSITIVITY_REL_GRID:
            if rg > t:
                sens[f"{t}"] += 1

    n = len(pairs)
    rep = sd_sources["reported"]
    tot_sides = sd_sources["reported"] + sd_sources["defaulted"]
    out = {
        "n_candidate_pairs": n,
        "beyond_noise_pairs_PRIMARY": beyond,
        "beyond_noise_rate": round(beyond / n, 3),
        "gate0_floor_for_reference": 300,
        "healthy_not_near_floor": beyond >= 600,
        "by_range_type_all": dict(by_range),
        "by_range_type_beyond": dict(by_range_beyond),
        "by_split_beyond": dict(by_split_beyond),
        "dispersion_source_share": {
            "reported_sides": rep, "defaulted_sides": tot_sides - rep,
            "reported_fraction": round(rep / tot_sides, 4),
        },
        "sensitivity_simple_rel_gap": sens,
        "example_beyond_pairs": beyond_pairs_ids,
    }
    outdir = RUNS / "phase2_noise"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "noise_viability.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

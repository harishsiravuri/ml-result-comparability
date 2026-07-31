"""Phase 2: run the FROZEN preregistered noise model (deterministic, $0).

Emits data/census/noise_decisions.jsonl (a beyond-noise decision plus evidence for EVERY
candidate; analysis-side, never enters a prompt, no Papers-with-Code gold value touched).
Reports the beyond-noise prevalence with Wilson confidence intervals on the DEV split
ONLY; the test prevalence is the single-shot finding (Phase 5) and is held back here.

Test decisions ARE computed (the rule is deterministic and the judge needs them in
Phase 3 and Phase 5), but no test prevalence is reported or analyzed in this phase.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.paths import CENSUS, RUNS  # noqa: E402
from census.surface import reconcile  # noqa: E402
from noise.model import K, SENSITIVITY_REL_GRID, decide  # noqa: E402
from noise.stats import wilson_interval  # noqa: E402


def _wilson_row(k: int, n: int) -> dict:
    p, lo, hi = wilson_interval(k, n)
    return {"n": n, "beyond": k, "prevalence": round(p, 4),
            "ci95": [round(lo, 4), round(hi, 4)]}


def main() -> None:
    pairs = [json.loads(l) for l in open(CENSUS / "candidates.jsonl") if l.strip()]
    decisions = []
    for p in pairs:
        v1, v2 = float(p["left"]["value"]), float(p["right"]["value"])
        x1, x2, adj = reconcile(v1, v2)
        d = decide(x1, x2, p["metric_canonical"] or p["metric_id"],
                   p["left"]["evidence_quote"], p["right"]["evidence_quote"])
        rec = {
            "pair_id": p["pair_id"], "split": p["split"], "pair_type": p["pair_type"],
            "identity_grade": p["identity_grade"], "task_family": p["task_family"],
            "metric_canonical": p["metric_canonical"],
            "beyond_noise": d.beyond_noise, "range_type": d.range_type,
            "gap": round(d.gap, 6), "threshold": round(d.threshold, 6),
            "decision_units": d.decision_units,
            "sigma_left": round(d.sigma_left, 6), "sigma_right": round(d.sigma_right, 6),
            "sd_source_left": d.sd_source_left, "sd_source_right": d.sd_source_right,
            "reconciled_values": [round(x1, 6), round(x2, 6)],
            "unit_scale_reconciled": adj,
        }
        decisions.append(rec)

    # write ALL decisions (deterministic; needed downstream)
    with open(CENSUS / "noise_decisions.jsonl", "w") as f:
        for r in decisions:
            f.write(json.dumps(r) + "\n")

    # ---- DEV-ONLY reporting ----
    dev = [r for r in decisions if r["split"] == "dev"]
    test = [r for r in decisions if r["split"] == "test"]

    def prevalence_by(key: str, rows: list) -> dict:
        groups = defaultdict(lambda: [0, 0])
        for r in rows:
            groups[r[key]][1] += 1
            if r["beyond_noise"]:
                groups[r[key]][0] += 1
        return {str(k): _wilson_row(v[0], v[1]) for k, v in sorted(groups.items(), key=lambda kv: -kv[1][1])}

    dev_beyond = sum(1 for r in dev if r["beyond_noise"])
    # reported-vs-defaulted dispersion share (dev)
    sd_sources = Counter()
    for r in dev:
        sd_sources[r["sd_source_left"]] += 1
        sd_sources[r["sd_source_right"]] += 1
    rep, deflt = sd_sources["reported"], sd_sources["defaulted"]
    # sensitivity (dev): simple relative-gap screen for comparability
    sens = {}
    for t in SENSITIVITY_REL_GRID:
        c = 0
        for r in dev:
            a, b = r["reconciled_values"]
            if abs(a - b) / max(abs(a), abs(b), 1e-9) > t:
                c += 1
        sens[f"{t}"] = c

    report = {
        "split_reported": "DEV ONLY (test held for Phase 5 single-shot)",
        "k": K,
        "dev": {
            "overall": _wilson_row(dev_beyond, len(dev)),
            "by_pair_type": prevalence_by("pair_type", dev),
            "by_identity_grade": prevalence_by("identity_grade", dev),
            "by_range_type": prevalence_by("range_type", dev),
            "by_task_family_top12": dict(list(prevalence_by("task_family", dev).items())[:12]),
            "dispersion_source_share": {
                "reported_sides": rep, "defaulted_sides": deflt,
                "reported_fraction": round(rep / max(rep + deflt, 1), 4),
            },
            "sensitivity_simple_rel_gap": sens,
        },
        "test_held": {"n_pairs": len(test),
                      "note": "deterministic decisions computed and stored; prevalence NOT reported until Phase 5"},
    }
    out = RUNS / "phase2_noise"
    out.mkdir(parents=True, exist_ok=True)
    (out / "dev_prevalence.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {CENSUS/'noise_decisions.jsonl'} ({len(decisions):,} decisions; "
          f"{len(dev):,} dev / {len(test):,} test)")


if __name__ == "__main__":
    main()

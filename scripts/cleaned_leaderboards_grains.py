"""Phase 6 addition: grain-specific cuts of the cleaned-leaderboards divergence
(deterministic, $0; reads cleaned_leaderboards.jsonl only). Reports winner-change and the
honest three-way pair split for: all leaderboards (conservative whole-corpus), all_pwc
(PRIMARY, well-specified), the demonstrably multi-protocol subset, and the other grades.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.paths import REPO_ROOT  # noqa: E402
from noise.stats import wilson_interval  # noqa: E402

D = REPO_ROOT / "data" / "cleaned_leaderboards"


def cut(lbs, name):
    kd = [l for l in lbs if l["metric_direction"] in ("higher", "lower")]
    wc = sum(1 for l in kd if l["winner_changed"])
    p, lo, hi = wilson_interval(wc, len(kd)) if kd else (None, None, None)
    return {
        "grain": name, "n_leaderboards": len(lbs), "n_known_direction": len(kd),
        "winner_change_rate": round(p, 4) if p is not None else None,
        "winner_change_ci95": [round(lo, 4), round(hi, 4)] if p is not None else None,
        "mean_pair_comparable": round(st.mean(l["pair_comparable_fraction"] for l in lbs), 4) if lbs else None,
        "mean_pair_confirmed_incomparable": round(st.mean(l["pair_confirmed_incomparable_fraction"] for l in lbs), 4) if lbs else None,
        "mean_pair_unknown": round(st.mean(l["pair_unknown_fraction"] for l in lbs), 4) if lbs else None,
        "mean_top3_not_confirmed_comparable": round(st.mean(
            l["top3_not_confirmed_comparable_to_principal"] for l in kd
            if l["top3_not_confirmed_comparable_to_principal"] is not None), 4) if kd else None,
        "mean_principal_cluster_coverage": round(st.mean(l["principal_cluster_coverage"] for l in lbs), 4) if lbs else None,
    }


def main():
    lbs = [json.loads(l) for l in open(D / "cleaned_leaderboards.jsonl") if l.strip()]
    grains = {
        "all_leaderboards (conservative whole-corpus)": lbs,
        "all_pwc (PRIMARY, well-specified)": [l for l in lbs if l["identity_grade"] == "all_pwc"],
        "demonstrably_multiprotocol (>=2 comparable clusters)": [l for l in lbs if l["n_comparable_clusters"] >= 2],
        "partial_pwc": [l for l in lbs if l["identity_grade"] == "partial_pwc"],
        "hash_only": [l for l in lbs if l["identity_grade"] == "hash_only"],
    }
    out = {"grains": [cut(v, k) for k, v in grains.items()]}
    (D / "divergence_by_grain.json").write_text(json.dumps(out, indent=2))
    print(f"{'grain':<52} {'n':>5} {'n_dir':>6} {'win_chg':>8} {'comp':>6} {'conf_inc':>9} {'unknown':>8}")
    for g in out["grains"]:
        print(f"{g['grain']:<52} {g['n_leaderboards']:>5} {g['n_known_direction']:>6} "
              f"{str(g['winner_change_rate']):>8} {str(g['mean_pair_comparable']):>6} "
              f"{str(g['mean_pair_confirmed_incomparable']):>9} {str(g['mean_pair_unknown']):>8}")


if __name__ == "__main__":
    main()

"""Item 2: emit comparability certificates (deterministic, $0) and validate self-consistency.

Scope (documented, not silent):
  - ALL 3,058 released census candidate pairs (complete coverage of the released dataset).
  - Leaderboard-pair certificates for the top-10 entries of each multi-protocol leaderboard,
    i.e. the comparisons that drive the winner-change finding. Emitting all 140,277
    adjudicable multi-protocol pairs would produce a multi-hundred-megabyte artifact, so the
    bound is stated here rather than applied silently.

Validation: status (what the resource reports) must equal implied_status (what per_facet plus
normalizations imply). Mismatches are reported and broken down by kind.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.paths import CENSUS, REPO_ROOT  # noqa: E402
from certificates.certificate import build_certificate  # noqa: E402
from certificates.facets import side_protocol  # noqa: E402

RELEASE = REPO_ROOT / "data" / "release"
TOP_K = 10


def census_certificates():
    cands = {json.loads(l)["pair_id"]: json.loads(l) for l in open(CENSUS / "candidates.jsonl") if l.strip()}
    noise = {json.loads(l)["pair_id"]: json.loads(l) for l in open(CENSUS / "noise_decisions.jsonl") if l.strip()}
    judge = {}
    for src in ("phase5_test_predictions.jsonl", "judge_dev.jsonl"):
        p = CENSUS / src
        if not p.exists():
            continue
        for l in open(p):
            r = json.loads(l)
            if r.get("method") == "judge_frontier" or r.get("backbone") == "frontier":
                judge.setdefault(r["pair_id"], r.get("final_cause", r.get("cause")))

    out = []
    for pid, p in cands.items():
        nd = noise.get(pid, {})
        norm = ([{"facet": "unit", "op": "percent_to_fraction_rescale", "side": "auto"}]
                if p.get("unit_scale_reconciled") else [])
        srcs = [nd.get("sd_source_left"), nd.get("sd_source_right")]
        disp = ("reported" if all(s == "reported" for s in srcs if s)
                else ("mixed" if "reported" in srcs else "default"))
        out.append(build_certificate(
            cert_id=pid, source="census_pair",
            cell={"method": p["method_canonical"], "dataset": p["dataset_canonical"],
                  "metric": p["metric_canonical"], "metric_direction": p.get("metric_direction")},
            left_rec={"arxiv_id": p["left"]["paper_id"], "source_location": p["left"].get("source_block"),
                      "value": float(p["left"]["value"])},
            right_rec={"arxiv_id": p["right"]["paper_id"], "source_location": p["right"].get("source_block"),
                       "value": float(p["right"]["value"])},
            left_proto=side_protocol(p["left"]), right_proto=side_protocol(p["right"]),
            normalizations=norm, judge_cause=judge.get(pid),
            uncertainty={"beyond_noise": nd.get("beyond_noise"), "dispersion_source": disp,
                         "confidence": None},
        ))
    return out


def leaderboard_certificates():
    lbs = [json.loads(l) for l in open(REPO_ROOT / "data" / "cleaned_leaderboards" / "cleaned_leaderboards.jsonl")]
    out = []
    for lb in lbs:
        if lb["n_comparable_clusters"] < 2:
            continue
        entries = []
        for c in lb["clusters"]:
            proto = {"split": c["protocol"]["split"], "metric_surface": c["protocol"]["metric_surface"],
                     "unit": c["protocol"]["unit"]}
            for e in c["ranking"]:
                entries.append((e, proto))
        rev = lb["metric_direction"] == "higher"
        entries.sort(key=lambda t: t[0]["value"], reverse=rev)
        for (a, pa), (b, pb) in combinations(entries[:TOP_K], 2):
            cid = f"{lb['leaderboard_id']}::{a['arxiv_id']}:{a['value']}::{b['arxiv_id']}:{b['value']}"
            out.append(build_certificate(
                cert_id=cid, source="leaderboard_pair",
                cell={"method": f"{a['method']} vs {b['method']}", "dataset": lb["dataset"],
                      "metric": lb["metric"], "metric_direction": lb["metric_direction"]},
                left_rec={"arxiv_id": a["arxiv_id"], "source_location": a.get("source_location"),
                          "value": float(a["value"])},
                right_rec={"arxiv_id": b["arxiv_id"], "source_location": b.get("source_location"),
                           "value": float(b["value"])},
                left_proto=pa, right_proto=pb, normalizations=[], judge_cause=None,
                uncertainty={"beyond_noise": None, "dispersion_source": None, "confidence": None},
            ))
    return out


def main():
    certs = census_certificates() + leaderboard_certificates()
    RELEASE.mkdir(parents=True, exist_ok=True)
    with open(RELEASE / "comparability_certificates.jsonl", "w") as f:
        for c in certs:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    by_src = Counter(c["source"] for c in certs)
    status = Counter(c["status"] for c in certs)
    typed = Counter(c["typed_outcome"] for c in certs)
    judged = [c for c in certs if c["judge_cause"]]
    non_excluded = [c for c in certs if c["status"] != "excluded"]
    mismatches = [c for c in certs if not c["self_consistent"]]
    mm_kinds = Counter(f'{c["status"]}|implied={c["implied_status"]}' for c in mismatches)

    # leakage guard: no excerpt text in any certificate
    text_keys = {"evidence_quote", "rationale", "note"}
    leaks = sum(1 for c in certs if text_keys & set(json.dumps(c).split('"')))

    summary = {
        "n_certificates": len(certs), "by_source": dict(by_src),
        "scope_note": "all released census pairs; leaderboard pairs limited to the top-%d "
                      "entries of each multi-protocol leaderboard (bound stated, not silent)" % TOP_K,
        "counts_by_status": dict(status),
        "counts_by_typed_outcome": dict(typed),
        "self_consistency": {
            "overall_pass_rate": round(sum(c["self_consistent"] for c in certs) / len(certs), 4),
            "pass_rate_excluding_excluded": round(
                sum(c["self_consistent"] for c in non_excluded) / max(len(non_excluded), 1), 4),
            "pass_rate_on_judge_traced": round(
                sum(c["self_consistent"] for c in judged) / max(len(judged), 1), 4),
            "n_judge_traced": len(judged),
            "n_mismatches": len(mismatches),
            "mismatch_kinds": dict(mm_kinds),
        },
        "evidence_text_leak_check": leaks,
    }
    (CENSUS / "certificates_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

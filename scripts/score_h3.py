"""H3 endpoint pilot, Step 2: compute the pre-committed analysis (docs/h3_analysis_plan.md).

Frozen at Step 1; run only after the author labels return. It computes EXACTLY the frozen list
and nothing else. No rerun, no restratification, no new analysis after seeing labels. Refuses to
run until the judgment sheet carries labels. Deterministic, no model call, $0.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from common.paths import CENSUS, RUNS  # noqa: E402

BOOT, SEED = 10000, 20260726
TRUE_PROP = {"multi_protocol": 190 / 3061, "remainder": 2871 / 3061}


def _rate(claims):
    """Unsupported rate over judged claims (cannot_judge excluded from num and denom)."""
    judged = [c for c in claims if c["judgment"] in ("supported", "unsupported")]
    if not judged:
        return None, 0, 0
    unsup = sum(1 for c in judged if c["judgment"] == "unsupported")
    return unsup / len(judged), unsup, len(judged)


def _reduction(claims):
    b, _, nb = _rate([c for c in claims if c["system"] == "blind"])
    q, _, nq = _rate([c for c in claims if c["system"] == "qualified"])
    if b is None or q is None:
        return None
    return b - q


def _boot_ci(per_board_claims, stat, stratified=False):
    rng = random.Random(SEED)
    boards = list(per_board_claims)
    draws, nulls = [], 0
    for _ in range(BOOT):
        if stratified:
            by = defaultdict(list)
            for b in boards:
                by[b["stratum"]].append(b)
            parts = {}
            ok = True
            for s, bs in by.items():
                res = [bs[rng.randrange(len(bs))] for _ in bs]
                r = _reduction([c for b in res for c in b["claims"]])
                if r is None:
                    ok = False
                    break
                parts[s] = r
            if not ok:
                nulls += 1
                continue
            val = sum(TRUE_PROP[s] * parts.get(s, 0.0) for s in TRUE_PROP)
        else:
            res = [boards[rng.randrange(len(boards))] for _ in boards]
            val = stat([c for b in res for c in b["claims"]])
            if val is None:
                nulls += 1
                continue
        draws.append(val)
    draws.sort()
    if not draws:
        return [None, None, None, nulls]
    return [round(draws[int(0.025 * len(draws))], 4),
            round(draws[int(0.975 * len(draws))], 4), nulls]


def main():
    labels = {r["claim_id"]: r["judgment"].strip()
              for r in csv.DictReader(open(CENSUS / "h3_judgment_sheet.csv"))}
    if not any(labels.values()):
        raise SystemExit("h3_judgment_sheet.csv is unlabeled; this is Step 2, run after labeling.")
    key = {json.loads(l)["claim_id"]: json.loads(l)
           for l in open(CENSUS / "h3_claim_key.jsonl") if l.strip()}
    coverage = [json.loads(l) for l in open(CENSUS / "h3_coverage_ledger.jsonl") if l.strip()]

    claims = []
    for cid, k in key.items():
        claims.append({**k, "judgment": labels.get(cid, "")})
    by_board = defaultdict(lambda: {"stratum": None, "claims": []})
    for c in claims:
        by_board[c["leaderboard_id"]]["stratum"] = c["stratum"]
        by_board[c["leaderboard_id"]]["claims"].append(c)
    boards = [{"leaderboard_id": k, **v} for k, v in by_board.items()]

    blind = [c for c in claims if c["system"] == "blind"]
    qual = [c for c in claims if c["system"] == "qualified"]
    b_rate, b_u, b_n = _rate(blind)
    q_rate, q_u, q_n = _rate(qual)

    n_sampled = len(boards) + 0  # boards holding definite claims
    n_boards_total = 60
    coverage_share = round(len(qual) / n_boards_total, 4)

    # matched secondary: leaderboards with a qualified definite claim
    matched_ids = {c["leaderboard_id"] for c in qual}
    matched = [b for b in boards if b["leaderboard_id"] in matched_ids]
    mb = [c for b in matched for c in b["claims"] if c["system"] == "blind"]
    mq = [c for b in matched for c in b["claims"] if c["system"] == "qualified"]
    mb_rate, _, mb_n = _rate(mb)
    mq_rate, _, mq_n = _rate(mq)

    prim = _boot_ci(boards, _reduction, stratified=False)
    post = _boot_ci(boards, _reduction, stratified=True)
    primary_reduction = (b_rate - q_rate) if (b_rate is not None and q_rate is not None) else None
    post_point = None
    if b_rate is not None and q_rate is not None:
        by_s = defaultdict(list)
        for b in boards:
            by_s[b["stratum"]].extend(b["claims"])
        red_s = {s: _reduction(cs) for s, cs in by_s.items()}
        if all(v is not None for v in red_s.values()):
            post_point = sum(TRUE_PROP[s] * red_s[s] for s in TRUE_PROP)

    out = {
        "provisional": True, "author_labeled": True,
        "independent_annotator_subset": "to follow",
        "n_leaderboards": n_boards_total,
        "cannot_judge_counts": {
            "blind": sum(1 for c in blind if c["judgment"] == "cannot_judge"),
            "qualified": sum(1 for c in qual if c["judgment"] == "cannot_judge")},
        "blind_unsupported_rate": {"rate": round(b_rate, 4) if b_rate is not None else None,
                                   "unsupported": b_u, "judged": b_n},
        "qualified_unsupported_rate": {"rate": round(q_rate, 4) if q_rate is not None else None,
                                       "unsupported": q_u, "judged": q_n},
        "coverage_qualified_definite_share": coverage_share,
        "primary_reduction_enriched_sample": {
            "note": "multi-protocol-ENRICHED sample; NOT the corpus rate",
            "reduction": round(primary_reduction, 4) if primary_reduction is not None else None,
            "clustered_bootstrap_ci95": prim[:2], "null_replicates": prim[2]},
        "post_stratified_corpus_reduction": {
            "note": "reweighted to true corpus proportions %s; corpus-representative" % TRUE_PROP,
            "reduction": round(post_point, 4) if post_point is not None else None,
            "clustered_bootstrap_ci95": post[:2], "null_replicates": post[2]},
        "matched_secondary_paired": {
            "note": "leaderboards where the qualified system also makes a definite-winner claim",
            "n_leaderboards": len(matched),
            "blind_unsupported_rate": round(mb_rate, 4) if mb_rate is not None else None,
            "qualified_unsupported_rate": round(mq_rate, 4) if mq_rate is not None else None,
            "blind_judged": mb_n, "qualified_judged": mq_n},
        "descriptive_changed_conclusion": {
            "note": "label-free context only; the blind winner differs from the qualified "
                    "winner. Pre-committed at Step 1 as 3 of the 29 qualified-definite boards.",
            "count": sum(1 for b in boards
                         if (lambda cl: "qualified" in cl
                             and cl["blind"]["claimed_winner"] != cl["qualified"]["claimed_winner"])
                         ({c["system"]: c for c in b["claims"]})),
            "n_qualified_definite_boards": sum(
                1 for b in boards if any(c["system"] == "qualified" for c in b["claims"]))},
        "decision_flags": {
            "primary_ci_excludes_zero": bool(prim[0] is not None and (prim[0] > 0 or prim[1] < 0)),
            "post_stratified_ci_excludes_zero": bool(post[0] is not None and (post[0] > 0 or post[1] < 0)),
            "rule": "H3 supported when the reduction is positive and the CI excludes zero; if "
                    "either CI includes zero, flag and stop (adjust proposal wording)."},
        "bootstrap": {"draws": BOOT, "seed": SEED, "unit": "leaderboard"},
    }
    (CENSUS / "h3_scores.json").write_text(json.dumps(out, indent=2))
    d = RUNS / "h3"
    d.mkdir(parents=True, exist_ok=True)
    (d / "h3_scores.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

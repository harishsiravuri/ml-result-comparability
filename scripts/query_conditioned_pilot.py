"""Item 4: query-conditioned comparability pilot (deterministic, $0).

Implements Q_any, Q_official, Q_trend and Q_threshold (framework doc section 3, registry in
src/certificates/query_templates.py) over the already-extracted protocol facets in
data/cleaned_leaderboards/cleaned_leaderboards.jsonl. No LLM calls.

Q-compatibility: records r, r' are Q-compatible when they agree on every facet Q marks
must-agree AND neither is missing such a facet. Valid comparison sets are therefore the
signature groups over the must-agree facets, plus a singleton for every record missing one of
those facets (a singleton is maximal because such a record is compatible with nothing).

Answer status for the argmax ("winner"):
  robust       - every valid comparison set with at least two records names the same winner;
  conditional  - the sets disagree on the winner;
  insufficient - no valid comparison set holds at least two records.

Q_threshold is a verification query built on Q_official: the champion M* and threshold tau (the
best competitor value) are fixed from the principal Q_official comparison set, and the answer is
robust when M* strictly exceeds tau in every full-protocol group that measures it against a peer.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from certificates.facets import strip_unit_annotation  # noqa: E402
from certificates.query_templates import must_agree  # noqa: E402
from common.paths import CENSUS, REPO_ROOT  # noqa: E402
from noise.stats import wilson_interval  # noqa: E402

BOOT, SEED = 4000, 4242
_FULL = ("split", "metric_surface", "unit")


def year_of(arxiv_id: str):
    s = str(arxiv_id).split(".")[0]
    if len(s) >= 2 and s[:2].isdigit():
        yy = int(s[:2])
        return 2000 + yy if yy < 50 else 1900 + yy
    return None


def entries_of(lb, normalize_annotation=False):
    """Every record on the leaderboard with its observed protocol (None facet = missing).

    normalize_annotation moves a trailing percent annotation off the metric surface and into
    the unit facet, so `miou (%)` and `miou` land in the same comparison set. It is OFF by
    default: the primary run must mirror the released cleaned-leaderboard clustering exactly.
    """
    def proto(split, ms, unit):
        if normalize_annotation:
            base, annot = strip_unit_annotation(ms)
            ms, unit = base, (unit or annot)
        return {"split": split, "metric_surface": ms, "unit": unit}

    out = []
    for c in lb["clusters"]:
        p = c["protocol"]
        for e in c["ranking"]:
            out.append({"method": e["method"], "value": float(e["value"]),
                        "arxiv_id": e["arxiv_id"],
                        "proto": proto(p["split"], p["metric_surface"], p["unit"])})
    for e in lb.get("comparability_unknown_entries", []):
        out.append({"method": e["method"], "value": float(e["value"]),
                    "arxiv_id": e.get("paper_id"),
                    "proto": {"split": None, "metric_surface": None, "unit": None}})
    return out


def winner(records, direction):
    if not records:
        return None
    r = (max if direction == "higher" else min)(records, key=lambda x: x["value"])
    return r["method"]


def comparison_sets(records, must_agree):
    """Valid comparison sets under the must-agree facet list."""
    if not must_agree:
        return [list(records)]                      # Q_any: one set containing everything
    groups, singles = defaultdict(list), []
    for r in records:
        if any(r["proto"].get(f) is None for f in must_agree):
            singles.append([r])                     # missing a must-agree facet -> singleton
        else:
            groups[tuple(r["proto"][f] for f in must_agree)].append(r)
    return list(groups.values()) + singles


def status_of(sets, direction):
    usable = [s for s in sets if len(s) >= 2]
    if not usable:
        return "insufficient", None
    winners = {winner(s, direction) for s in usable}
    if len(winners) == 1:
        return "robust", next(iter(winners))
    return "conditional", None


def _best_value(records, direction):
    return (max if direction == "higher" else min)(r["value"] for r in records)


def _exceeds(value, tau, direction):
    return value > tau if direction == "higher" else value < tau


def full_protocol_groups(records):
    """Groups keyed by the complete observed protocol signature; records missing any facet are
    dropped, since a protocol-homogeneous group requires all three facets observed."""
    groups = defaultdict(list)
    for r in records:
        sig = tuple(r["proto"].get(f) for f in _FULL)
        if all(v is not None for v in sig):
            groups[sig].append(r)
    return groups


def q_threshold_status(records, direction):
    """Q_threshold: does the official champion's lead over the runner-up survive protocol
    conditioning? Returns (status, champion, detail)."""
    # reference = principal Q_official comparison set: split=test, group by (metric_surface,
    # unit), at least two DISTINCT methods, largest (deterministic tie-break by signature).
    official = defaultdict(list)
    for r in records:
        if r["proto"].get("split") == "test":
            ms, un = r["proto"].get("metric_surface"), r["proto"].get("unit")
            if ms is not None and un is not None:
                official[(ms, un)].append(r)
    refs = [(sig, g) for sig, g in official.items() if len({x["method"] for x in g}) >= 2]
    if not refs:
        return "insufficient", None, {"reason": "no principal Q_official comparison set"}
    sig0, ref = sorted(refs, key=lambda kv: (-len(kv[1]), kv[0]))[0]
    champ = winner(ref, direction)
    competitors = [r for r in ref if r["method"] != champ]
    if not competitors:
        return "insufficient", None, {"reason": "champion has no runner-up in the reference set"}
    tau = _best_value(competitors, direction)

    # test groups: full-protocol groups where M* is measured against at least one other method.
    tests = [g for g in full_protocol_groups(records).values()
             if champ in {x["method"] for x in g} and len({x["method"] for x in g}) >= 2]
    if len(tests) < 2:
        return "insufficient", champ, {
            "reason": "champion measured against a peer under only one protocol",
            "threshold_tau": tau, "n_test_groups": len(tests)}

    passes = [_exceeds(_best_value([x for x in g if x["method"] == champ], direction), tau,
                       direction) for g in tests]
    detail = {"threshold_tau": tau, "n_test_groups": len(tests),
              "n_groups_clearing_threshold": sum(passes)}
    if all(passes):
        return "robust", champ, detail
    return "conditional", champ, detail


def score(lbs, normalize_annotation=False):
    rows, agg = [], defaultdict(Counter)
    flips = []          # per multi-protocol leaderboard: 1 if Q_any winner flips vs robust Q_official
    for lb in lbs:
        d = lb["metric_direction"]
        if d not in ("higher", "lower"):
            continue
        recs = entries_of(lb, normalize_annotation)
        if len(recs) < 2:
            continue
        if normalize_annotation:
            # recompute multi-protocol from the normalized signatures: merging an annotated
            # surface with its plain twin can drop a board below two observed protocols
            sigs = {tuple(r["proto"][f] for f in ("split", "metric_surface", "unit"))
                    for r in recs if all(r["proto"][f] for f in ("split", "metric_surface", "unit"))}
            multi = len(sigs) >= 2
        else:
            multi = lb["n_comparable_clusters"] >= 2

        # Q_any: must-agree nothing -> the naive single comparison set
        any_sets = comparison_sets(recs, [])
        any_status, any_winner = status_of(any_sets, d)
        if any_winner is None and any_sets and len(any_sets[0]) >= 2:
            any_winner = winner(any_sets[0], d)

        # Q_official: restrict to the official/test split, then must-agree metric_surface+unit
        official_recs = [r for r in recs if r["proto"].get("split") == "test"]
        off_sets = comparison_sets(official_recs, must_agree("Q_official"))
        off_status, off_winner = status_of(off_sets, d)

        # Q_threshold: does the official champion's lead survive protocol conditioning?
        thr_status, thr_champ, thr_detail = q_threshold_status(recs, d)

        # Q_trend: must-agree split+metric_surface+unit, partitioned by publication year
        tr_sets = comparison_sets(recs, must_agree("Q_trend"))
        per_year = defaultdict(set)
        for s in tr_sets:
            by_year = defaultdict(list)
            for r in s:
                y = year_of(r["arxiv_id"])
                if y:
                    by_year[y].append(r)
            for y, rs in by_year.items():
                if len(rs) >= 2:
                    per_year[y].add(winner(rs, d))
        if not per_year:
            tr_status = "insufficient"
        elif all(len(v) == 1 for v in per_year.values()):
            tr_status = "robust"
        else:
            tr_status = "conditional"

        agg["Q_any"][any_status] += 1
        agg["Q_official"][off_status] += 1
        agg["Q_trend"][tr_status] += 1
        agg["Q_threshold"][thr_status] += 1

        flipped = (off_status == "robust" and any_winner is not None
                   and off_winner is not None and any_winner != off_winner)
        rows.append({
            "leaderboard_id": lb["leaderboard_id"], "dataset": lb["dataset"],
            "metric": lb["metric"], "metric_direction": d,
            "identity_grade": lb["identity_grade"],
            "multi_protocol": multi, "n_records": len(recs),
            "existing_winner_changed": lb["winner_changed"],
            "Q_any": {"status": any_status, "winner": any_winner, "n_sets": len(any_sets)},
            "Q_official": {"status": off_status, "winner": off_winner,
                           "n_sets": len(off_sets), "n_official_records": len(official_recs)},
            "Q_trend": {"status": tr_status, "n_years_answerable": len(per_year)},
            "Q_threshold": {"status": thr_status, "champion": thr_champ, **thr_detail},
            "winner_flips_any_vs_official": bool(flipped),
            "threshold_conditional": thr_status == "conditional",
        })
        if multi:
            flips.append(1.0 if flipped else 0.0)

    rng = random.Random(SEED)

    def boot(vals):
        if not vals:
            return [None, None, None]
        pt = sum(vals) / len(vals)
        bs = []
        for _ in range(BOOT):
            s = [vals[rng.randrange(len(vals))] for _ in vals]
            bs.append(sum(s) / len(s))
        bs.sort()
        return [round(pt, 4), round(bs[int(0.025 * BOOT)], 4), round(bs[int(0.975 * BOOT)], 4)]

    def rate(vals, definition):
        k, n = int(sum(vals)), len(vals)
        p, lo, hi = wilson_interval(k, n) if n else (None, None, None)
        return {"definition": definition, "k": k, "n": n,
                "rate": round(p, 4) if p is not None else None,
                "wilson_ci95": [round(lo, 4), round(hi, 4)] if p is not None else None,
                "bootstrap_ci95": boot(vals)[1:]}

    mp = [r for r in rows if r["multi_protocol"]]
    # restricted denominator: multi-protocol boards where Q_official is answerable AND robust
    answerable = [r for r in mp if r["Q_official"]["status"] == "robust"]
    flips_answerable = [1.0 if r["winner_flips_any_vs_official"] else 0.0 for r in answerable]
    pwc = [r for r in rows if r["identity_grade"] == "all_pwc"]

    # Q_threshold: conditional fraction among boards where it is answerable (not insufficient),
    # both whole-corpus and restricted to the demonstrably multi-protocol boards.
    thr_answerable = [r for r in rows if r["Q_threshold"]["status"] != "insufficient"]
    thr_answerable_mp = [r for r in mp if r["Q_threshold"]["status"] != "insufficient"]

    demo = {
        "what_this_is": "Deterministic query-conditioned pilot over the cleaned leaderboards. "
                        "No LLM calls; no use of the frozen census gold. Registry: "
                        "schemas/query_templates.json. Q_any must-agree {}, Q_official "
                        "restricts to split=test then must-agrees {metric_surface, unit}, "
                        "Q_trend must-agrees {split, metric_surface, unit} and partitions by "
                        "arXiv publication year, Q_threshold fixes the champion and the "
                        "runner-up threshold from the principal Q_official set and tests "
                        "whether the champion clears that threshold under every full protocol "
                        "in which it meets a peer.",
        "template_registry": "schemas/query_templates.json (schema_version 1.0.0)",
        "n_leaderboards_total": 4438,
        "n_leaderboards_scored": len(rows),
        "excluded": {"unknown_metric_direction_or_under_2_records": 4438 - len(rows)},
        "n_multi_protocol_scored": len(mp),
        "n_multi_protocol_total": 215,
        "status_distribution_by_query": {q: dict(c) for q, c in agg.items()},
        "structural_note_Q_any": "Q_any is robust on every leaderboard BY CONSTRUCTION: with no "
                                 "must-agree facet there is exactly one comparison set, so the "
                                 "naive query can never report uncertainty. That is the failure "
                                 "mode the query-conditioned framing targets, not a finding.",
        "headline_conditional_fraction": rate(
            flips_answerable,
            "PRIMARY: of multi-protocol leaderboards where Q_official is answerable AND robust, "
            "the fraction whose Q_any winner differs from the Q_official winner (the answer is "
            "conditional on the query even though it is robust within the query)"),
        "conditional_fraction_all_multi_protocol": rate(
            flips,
            "SECONDARY, wider denominator: of ALL scored multi-protocol leaderboards, the "
            "fraction with Q_official robust AND a Q_any/Q_official winner flip. Boards where "
            "Q_official is insufficient count in the denominator and never in the numerator."),
        "conditional_fraction_whole_corpus": rate(
            [1.0 if r["winner_flips_any_vs_official"] else 0.0
             for r in rows if r["Q_official"]["status"] == "robust"],
            "conservative whole-corpus: same PRIMARY measure over ALL scored leaderboards with a "
            "robust Q_official, not only the demonstrably multi-protocol ones"),
        "conditional_fraction_all_pwc_primary_grain": rate(
            [1.0 if r["winner_flips_any_vs_official"] else 0.0
             for r in pwc if r["Q_official"]["status"] == "robust"],
            "same PRIMARY measure restricted to the all_pwc (well-specified identity) grain"),
        "q_official_answerability_multi_protocol": dict(
            Counter(r["Q_official"]["status"] for r in mp)),
        "q_official_internally_conditional": rate(
            [1.0 if r["Q_official"]["status"] == "conditional" else 0.0 for r in mp],
            "of scored multi-protocol leaderboards, the fraction where Q_official is itself "
            "conditional: its own valid comparison sets disagree on the winner"),
        "q_threshold_conditional_fraction_answerable_multi_protocol": rate(
            [1.0 if r["Q_threshold"]["status"] == "conditional" else 0.0
             for r in thr_answerable_mp],
            "PRIMARY threshold measure: of multi-protocol leaderboards where Q_threshold is "
            "answerable (a champion measured against a peer under >=2 protocols), the fraction "
            "whose margin is conditional: the official champion clears the runner-up threshold "
            "under some protocols but not all"),
        "q_threshold_conditional_fraction_over_all_multi_protocol": rate(
            [1.0 if r["Q_threshold"]["status"] == "conditional" else 0.0 for r in mp],
            "same conditional count over ALL scored multi-protocol leaderboards (the "
            "insufficient boards, where the champion is not re-measured, count in the "
            "denominator): the unconditional prevalence of a protocol-dependent margin"),
        "q_threshold_conditional_fraction_answerable_whole_corpus": rate(
            [1.0 if r["Q_threshold"]["status"] == "conditional" else 0.0 for r in thr_answerable],
            "conservative whole-corpus threshold measure: same, over ALL scored leaderboards "
            "where Q_threshold is answerable, not only the demonstrably multi-protocol ones"),
        "q_threshold_answerability": {
            "n_answerable_whole_corpus": len(thr_answerable),
            "n_answerable_multi_protocol": len(thr_answerable_mp),
            "note": "Q_threshold is insufficient on most boards because the official champion "
                    "is usually measured against a peer under only one protocol; the "
                    "answerable subset is exactly the boards where a cross-protocol margin "
                    "test is possible."},
        "cross_check_vs_existing_winner_change": {
            "note": "divergence_by_grain.json reports winner_change_rate 0.4632 on the same 190 "
                    "scored multi-protocol boards. That measure compares the naive winner with "
                    "the principal-CLUSTER winner and fires on any protocol restriction; the "
                    "pilot measure is stricter, requiring the official-query answer to be robust "
                    "across every official comparison set. The two are not the same quantity.",
            "existing_winner_changed_rate_on_scored_multi_protocol": round(
                sum(1 for r in mp if r["existing_winner_changed"]) / len(mp), 4) if mp else None,
        },
        "intervals": {"wilson": "matches divergence_by_grain.json, which uses Wilson not "
                                "bootstrap; the bootstrap is reported alongside for parity with "
                                "the brief",
                      "bootstrap": {"draws": BOOT, "seed": SEED, "unit": "leaderboard"}},
    }
    return demo, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sensitivity", action="store_true")
    args = ap.parse_args()

    lbs = [json.loads(l) for l in
           open(REPO_ROOT / "data" / "cleaned_leaderboards" / "cleaned_leaderboards.jsonl")]

    demo, rows = score(lbs, normalize_annotation=False)
    with open(CENSUS / "query_conditioned_by_leaderboard.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    if not args.no_sensitivity:
        # Sensitivity ONLY. The released cleaned leaderboards cluster on the raw metric
        # surface, so 166 of 2,537 clusters carry a percent annotation that splits them from
        # their plain twin. The released resource is tagged and must not be rebuilt here, so
        # the corrected view is reported as a sensitivity rather than applied.
        alt, _ = score(lbs, normalize_annotation=True)
        keys = ["headline_conditional_fraction", "conditional_fraction_all_multi_protocol",
                "conditional_fraction_whole_corpus", "q_official_internally_conditional",
                "q_threshold_conditional_fraction_answerable_multi_protocol",
                "q_threshold_conditional_fraction_over_all_multi_protocol",
                "q_threshold_conditional_fraction_answerable_whole_corpus"]
        demo["sensitivity_unit_annotation_normalized"] = {
            "what": "re-scores the pilot with the percent annotation moved from the metric "
                    "surface to the unit facet (the item-2/3 correctness fix), WITHOUT "
                    "rebuilding the released cleaned-leaderboard resource",
            "n_multi_protocol_scored": alt["n_multi_protocol_scored"],
            "status_distribution_by_query": alt["status_distribution_by_query"],
            **{k: {"primary": [demo[k]["k"], demo[k]["n"], demo[k]["rate"]],
                   "normalized": [alt[k]["k"], alt[k]["n"], alt[k]["rate"]],
                   "wilson_ci95_normalized": alt[k]["wilson_ci95"]} for k in keys},
        }

    (CENSUS / "query_conditioned_demo.json").write_text(json.dumps(demo, indent=2))
    print(json.dumps(demo, indent=2))


if __name__ == "__main__":
    main()

"""H3 endpoint pilot, Step 1: deterministic sample + claim extraction (NO LLM, $0).

H3 endpoint: the rate of UNSUPPORTED winner claims before and after protocol qualification,
judged against adjudicated author judgment. This script draws the frozen sample, extracts the
two claims per leaderboard deterministically, and writes a BLIND author judgment sheet. It does
NOT compute any rate: the unsupported rates need the author labels, which do not exist yet.

Two claims per leaderboard:
  - protocol-blind claim  = the naive Q_any top-1 winner (a definite-winner claim, always made);
  - protocol-qualified conclusion = the Q_official output: a definite winner (Q_official robust),
    a conditional winner (Q_official conditional), or an abstention with cause (insufficient).
Only definite-winner claims are judged; conditional and abstention outputs make no winner claim
and count as coverage, not claims.

The frozen census gold is used ONLY as displayed prior evidence where it covers a claim's
deciding pair; it never tunes the sample or the extraction and never substitutes for the new
author judgment.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from common.paths import CENSUS  # noqa: E402

# reuse the pilot's exact query logic so the claims match the released pilot
_spec = importlib.util.spec_from_file_location("qcp", REPO / "scripts" / "query_conditioned_pilot.py")
qcp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qcp)

SEED = 20260726
N_MP, N_REM = 40, 20
EVIDENCE_CAP = 15
CLEANED = REPO / "data" / "cleaned_leaderboards" / "cleaned_leaderboards.jsonl"


def entries_full(lb):
    """Records with full display fields (method, value, arxiv_id, source_location, proto)."""
    out = []
    for c in lb["clusters"]:
        p = c["protocol"]
        for e in c["ranking"]:
            out.append({"method": e["method"], "value": float(e["value"]),
                        "arxiv_id": e["arxiv_id"], "source_location": e.get("source_location"),
                        "proto": {"split": p["split"], "metric_surface": p["metric_surface"],
                                  "unit": p["unit"]}})
    for e in lb.get("comparability_unknown_entries", []):
        out.append({"method": e["method"], "value": float(e["value"]),
                    "arxiv_id": e.get("paper_id"), "source_location": None,
                    "proto": {"split": None, "metric_surface": None, "unit": None}})
    return out


def runner_up(records, winner_method, direction):
    """Best record whose method differs from the winner (the deciding competitor)."""
    others = [r for r in records if r["method"] != winner_method]
    if not others:
        return None
    return (max if direction == "higher" else min)(others, key=lambda r: r["value"])


def evidence_rows(records, direction, claimed, cap=EVIDENCE_CAP):
    """Top-cap records by value (always including the claimed winner), for display."""
    ordered = sorted(records, key=lambda r: r["value"], reverse=(direction == "higher"))
    keep = ordered[:cap]
    if not any(r["method"] == claimed for r in keep):
        champ = next((r for r in ordered if r["method"] == claimed), None)
        if champ:
            keep = keep[:cap - 1] + [champ]
    return [{"method": r["method"], "value": r["value"],
             "split": r["proto"]["split"], "metric_surface": r["proto"]["metric_surface"],
             "unit": r["proto"]["unit"], "arxiv_id": r["arxiv_id"],
             "source_location": r["source_location"]} for r in keep]


def official_winning_set(recs, direction):
    """The principal Q_official comparison set (test split, matched metric_surface+unit)."""
    official = [r for r in recs if r["proto"].get("split") == "test"]
    sets = qcp.comparison_sets(official, qcp.must_agree("Q_official"))
    usable = [s for s in sets if len(s) >= 2]
    if not usable:
        return []
    return max(usable, key=len)


def build_gold_prior_lookup():
    cands = {json.loads(l)["pair_id"]: json.loads(l)
             for l in open(CENSUS / "candidates.jsonl") if l.strip()}
    gold = {r["pair_id"]: r["label"].strip()
            for r in csv.DictReader(open(CENSUS / "gold_annotation_sheet.csv"))}
    lut = {}
    for pid, label in gold.items():
        p = cands[pid]
        k = (p["dataset_canonical"], p["metric_canonical"],
             frozenset((p["left"]["paper_id"], p["right"]["paper_id"])))
        lut[k] = label
    return lut


def main():
    rng = random.Random(SEED)
    scored = [json.loads(l) for l in open(CENSUS / "query_conditioned_by_leaderboard.jsonl")]
    cleaned = {json.loads(l)["leaderboard_id"]: json.loads(l) for l in open(CLEANED)}
    gold_lut = build_gold_prior_lookup()

    mp = sorted([r for r in scored if r["multi_protocol"]], key=lambda r: r["leaderboard_id"])
    rem = sorted([r for r in scored if not r["multi_protocol"]], key=lambda r: r["leaderboard_id"])
    true_prop = {"multi_protocol": len(mp) / len(scored), "remainder": len(rem) / len(scored)}

    sample = ([(r, "multi_protocol") for r in rng.sample(mp, N_MP)]
              + [(r, "remainder") for r in rng.sample(rem, N_REM)])
    sample.sort(key=lambda t: t[0]["leaderboard_id"])

    definite_claims = []           # each judged blind
    coverage_ledger = []           # conditional / abstention outputs (not judged)
    per_board = []
    n_gold_prior = 0

    for row, stratum in sample:
        lb = cleaned[row["leaderboard_id"]]
        d = row["metric_direction"]
        recs = entries_full(lb)
        dataset, metric = row["dataset"], row["metric"]

        def gold_prior(winner_method, records):
            ru = runner_up(records, winner_method, d)
            if ru is None:
                return None, None
            wa = next((r["arxiv_id"] for r in records if r["method"] == winner_method), None)
            k = (dataset, metric, frozenset((wa, ru["arxiv_id"])))
            return gold_lut.get(k), ru["method"]

        # ---- protocol-blind claim: Q_any top-1 winner (always a definite claim)
        blind_w = row["Q_any"]["winner"]
        b_prior, b_runnerup = gold_prior(blind_w, recs)
        n_gold_prior += 1 if b_prior else 0
        definite_claims.append({
            "system": "blind", "leaderboard_id": row["leaderboard_id"], "stratum": stratum,
            "dataset": dataset, "metric": metric, "metric_direction": d,
            "claimed_winner": blind_w, "deciding_runner_up": b_runnerup,
            "evidence": evidence_rows(recs, d, blind_w),
            "gold_prior": b_prior})

        # ---- protocol-qualified conclusion: Q_official output
        off = row["Q_official"]
        if off["status"] == "robust":
            win_set = official_winning_set(recs, d)
            q_prior, q_runnerup = gold_prior(off["winner"], win_set or recs)
            n_gold_prior += 1 if q_prior else 0
            definite_claims.append({
                "system": "qualified", "leaderboard_id": row["leaderboard_id"],
                "stratum": stratum, "dataset": dataset, "metric": metric,
                "metric_direction": d, "claimed_winner": off["winner"],
                "deciding_runner_up": q_runnerup,
                "evidence": evidence_rows(win_set or recs, d, off["winner"]),
                "gold_prior": q_prior})
            qualified_kind = "definite"
        elif off["status"] == "conditional":
            coverage_ledger.append({"leaderboard_id": row["leaderboard_id"], "stratum": stratum,
                                    "dataset": dataset, "metric": metric,
                                    "qualified_output": "conditional",
                                    "cause": "the official comparison sets disagree on the winner"})
            qualified_kind = "conditional"
        else:
            coverage_ledger.append({"leaderboard_id": row["leaderboard_id"], "stratum": stratum,
                                    "dataset": dataset, "metric": metric,
                                    "qualified_output": "abstention",
                                    "cause": "no official comparison set of at least two records"})
            qualified_kind = "abstention"

        per_board.append({
            "leaderboard_id": row["leaderboard_id"], "stratum": stratum,
            "dataset": dataset, "metric": metric, "metric_direction": d,
            "blind_winner": blind_w,
            "qualified_kind": qualified_kind,
            "qualified_winner": off["winner"] if qualified_kind == "definite" else None,
            "changed_conclusion": bool(qualified_kind == "definite" and off["winner"] != blind_w),
        })

    # ---- blind, order-randomized, interleaved presentation of the definite-winner claims.
    # Round-robin by leaderboard (group order and within-group order both seeded), so the two
    # claims from one leaderboard are separated by every other leaderboard's claims and cannot
    # sit adjacent. This interleaves the two systems and randomizes order at once.
    from collections import defaultdict as _dd
    groups = _dd(list)
    for c in definite_claims:
        groups[c["leaderboard_id"]].append(c)
    for g in groups.values():
        rng.shuffle(g)
    order = list(groups.keys())
    rng.shuffle(order)
    ordered = []
    while any(groups[k] for k in order):
        rng.shuffle(order)
        for k in order:
            if groups[k]:
                ordered.append(groups[k].pop())
    definite_claims = ordered
    for i, c in enumerate(definite_claims, 1):
        c["claim_id"] = f"h3c{i:03d}"

    # ---- write the hidden un-blinding key (NOT shown to the author)
    with open(CENSUS / "h3_claim_key.jsonl", "w") as f:
        for c in definite_claims:
            f.write(json.dumps({"claim_id": c["claim_id"], "system": c["system"],
                                "leaderboard_id": c["leaderboard_id"], "stratum": c["stratum"],
                                "claimed_winner": c["claimed_winner"],
                                "deciding_runner_up": c["deciding_runner_up"],
                                "has_gold_prior": bool(c["gold_prior"])}) + "\n")

    # ---- write the blind judgment sheet (markdown) + the blank CSV
    lines = ["# H3 judgment sheet (blind) — %d definite-winner claims" % len(definite_claims),
             "",
             "Each item is a claim that one method is best on a benchmark for a metric, together "
             "with the records behind it and each record's stated conditions. You are NOT told "
             "which system produced a claim, and there are no model predictions. For each claim, "
             "decide whether the claim is SUPPORTED by the records under the conditions a valid "
             "superiority claim requires (same split, same metric variant, same unit).",
             "",
             "Label each claim in `data/census/h3_judgment_sheet.csv` with ONE of: `supported`, "
             "`unsupported` (the records are incomparable under the required conditions, or the "
             "claim does not follow from the sources), `cannot_judge`. See "
             "`docs/h3_analysis_plan.md`.",
             ""]
    for c in definite_claims:
        lines += [
            "## %s" % c["claim_id"],
            "Claim: **%s** is the best method on **%s** for metric **%s** (%s is better)."
            % (c["claimed_winner"], c["dataset"], c["metric"],
               "higher" if c["metric_direction"] == "higher" else "lower"),
            "",
            "Records considered:",
            "",
            "| method | value | split | metric surface | unit | arxiv |",
            "|---|---|---|---|---|---|"]
        for r in c["evidence"]:
            lines.append("| %s | %s | %s | %s | %s | %s |" % (
                r["method"], r["value"], r["split"] or "unstated",
                r["metric_surface"] or "unstated", r["unit"] or "unstated",
                (r["arxiv_id"] or "") + (":" + r["source_location"] if r["source_location"] else "")))
        if c["gold_prior"]:
            lines += ["", "> Prior human comparability judgment exists for the "
                      "%s vs %s comparison on this benchmark: `%s` (prior evidence only; make "
                      "your own judgment)." % (c["claimed_winner"], c["deciding_runner_up"],
                                               c["gold_prior"])]
        lines += ["", "label: ____________  note: ____________", "", "---", ""]
    (CENSUS / "h3_judgment_sheet.md").write_text("\n".join(lines))

    with open(CENSUS / "h3_judgment_sheet.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["claim_id", "judgment", "note"])
        for c in definite_claims:
            w.writerow([c["claim_id"], "", ""])

    with open(CENSUS / "h3_coverage_ledger.jsonl", "w") as f:
        for r in coverage_ledger:
            f.write(json.dumps(r) + "\n")
    with open(CENSUS / "h3_sample.jsonl", "w") as f:
        for r in per_board:
            f.write(json.dumps(r) + "\n")

    n_blind = sum(1 for c in definite_claims if c["system"] == "blind")
    n_qual_def = sum(1 for c in definite_claims if c["system"] == "qualified")
    n_cond = sum(1 for r in coverage_ledger if r["qualified_output"] == "conditional")
    n_abst = sum(1 for r in coverage_ledger if r["qualified_output"] == "abstention")
    n_boards = len(sample)
    summary = {
        "frozen": True, "seed": SEED, "spend_usd": 0.0, "step": "1 (design and freeze)",
        "n_leaderboards_sampled": n_boards,
        "per_stratum_counts": {"multi_protocol": N_MP, "remainder": N_REM},
        "true_corpus_proportions": true_prop,
        "n_definite_winner_claims_total": len(definite_claims),
        "n_blind_claims": n_blind, "n_qualified_definite_claims": n_qual_def,
        "n_conditional": n_cond, "n_abstention": n_abst,
        "n_definite_by_stratum": {
            s: {"blind": sum(1 for c in definite_claims if c["stratum"] == s and c["system"] == "blind"),
                "qualified_definite": sum(1 for c in definite_claims if c["stratum"] == s and c["system"] == "qualified")}
            for s in ("multi_protocol", "remainder")},
        "label_free_descriptives": {
            "coverage_qualified_definite_share": round(n_qual_def / n_boards, 4),
            "coverage_note": "share of sampled leaderboards on which the qualified system makes "
                             "a definite-winner claim (1 minus the conditional-plus-abstention "
                             "share); this is label-free",
            "changed_conclusion_count_where_qualified_definite": sum(
                1 for r in per_board if r["changed_conclusion"]),
            "changed_conclusion_note": "descriptive only; the blind winner differs from the "
                                       "qualified winner (context, not the H3 endpoint)"},
        "n_claims_with_gold_prior": n_gold_prior,
        "unsupported_rates_status": "NOT COMPUTED — the blind and qualified unsupported rates "
                                    "require the author labels (Step 2). This file records only "
                                    "the frozen design and the label-free descriptives.",
        "artifacts": {
            "sample": "data/census/h3_sample.jsonl",
            "judgment_sheet_md": "data/census/h3_judgment_sheet.md",
            "judgment_sheet_csv": "data/census/h3_judgment_sheet.csv (blank)",
            "claim_key_hidden": "data/census/h3_claim_key.jsonl (NOT shown to the author)",
            "coverage_ledger": "data/census/h3_coverage_ledger.jsonl",
            "analysis_plan": "docs/h3_analysis_plan.md"},
    }
    (CENSUS / "h3_sample_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

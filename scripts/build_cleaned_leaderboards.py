"""Phase 6 ADDITION: comparability-cleaned leaderboards (deterministic, $0).

Uses EXISTING frozen outputs only (the tuple index + the canonicalization). Does NOT re-run,
re-tune, or re-grade the preregistered single-shot detector test, and adds no new frozen-test
evaluation. The cleaning applies the VALIDATED comparability grain (the protocol-vs-other
decision via the explicit, extractable protocol facets), NOT the weak fine-cause attribution,
and is conservative: it groups entries only on positive shared-protocol evidence and FLAGS
everything else as comparability-unknown (prefer flagging over false grouping).

Per canonical (dataset, metric) leaderboard:
  1. gather entries (one provenance-best row per method+paper), with value, span, identity grade;
  2. QUARANTINE extraction artifacts (critic_verdict UNSUPPORTED -> value not supported);
  3. CLUSTER comparable entries by the explicit protocol facets (normalized split family +
     raw metric surface variant + unit), all-known-and-equal; entries missing any facet go to
     a comparability-unknown bucket (not grouped with others);
  4. CLEANED ranking = within-cluster value ranking (metric direction aware); the principal
     cluster is the largest comparable cluster (the dominant protocol).

Naive-vs-cleaned divergence is reported descriptively (no model bar): top-1 winner change,
top-3 contamination, the fraction of head-to-head pairs that are cross-protocol, and the
Kendall tau over comparable pairs (which is ~1 by construction; see the report), all with
coverage and an identity-grade breakdown.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.paths import EXTRACTIONS, INDEX, REPO_ROOT  # noqa: E402
from common.metric_direction import metric_direction  # noqa: E402
from census.surface import reconcile, _norm  # noqa: E402
from judge.rules import _split_family  # noqa: E402

OUT = REPO_ROOT / "data" / "cleaned_leaderboards"
_CRITIC_RANK = {"SUPPORTED": 3, "PARTIAL": 2, None: 1, "": 1, "UNSUPPORTED": 0}


def _prov_key(r):
    cv = _CRITIC_RANK.get(r.critic_verdict if isinstance(r.critic_verdict, str) else "", 1)
    return (1 if bool(r.is_own_result) else 0, 1 if bool(r.quote_verified) else 0, cv,
            float(r.self_consistency) if pd.notna(r.self_consistency) else 0.0)


def identity_grade(mid, did, met):
    nh = [not str(x).startswith("hash:") for x in (mid, did, met)]
    return "all_pwc" if all(nh) else ("partial_pwc" if any(nh) else "hash_only")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(EXTRACTIONS / "tuples.parquet")
    df = df[df["value"].map(lambda v: pd.notna(v) and math.isfinite(float(v)))].copy()
    df["value"] = df["value"].astype(float)
    n_all = len(df)
    quarantined = int((df["critic_verdict"] == "UNSUPPORTED").sum())
    df = df[df["critic_verdict"] != "UNSUPPORTED"].copy()  # quarantine extraction errors

    import json as _json, gzip
    canon = _json.load(gzip.open(INDEX / "canon_tables.json.gz", "rt"))["entries"]

    def cname(cid, fb):
        e = canon.get(str(cid))
        return e["canonical_name"] if e else str(fb)

    df["lb"] = list(zip(df["dataset_id"], df["metric_id"]))

    leaderboards = []
    agg = {"n_leaderboards": 0, "winner_changed": 0, "known_direction": 0,
           "sum_comparable": 0.0, "sum_confirmed_incomp": 0.0, "sum_unknown": 0.0,
           "sum_top3_contam": 0.0, "n_top3": 0, "n_multicluster": 0, "sum_principal_cov": 0.0,
           "by_grade": defaultdict(lambda: {"n": 0, "winner_changed": 0})}

    for (did, mid), g in df.groupby("lb", sort=True):
        # one provenance-best entry per (method_id, paper_id)
        entries = []
        for (meth_id, pid), mg in g.groupby(["method_id", "paper_id"], sort=True):
            best = max(mg.itertuples(index=False), key=_prov_key)
            entries.append(best)
        # distinct methods needed for a ranking
        methods = {e.method_id for e in entries}
        if len(entries) < 2 or len(methods) < 2:
            continue
        metric_name = cname(mid, g["metric"].iloc[0])
        direction = metric_direction(str(metric_name))
        ds_name = cname(did, g["dataset"].iloc[0])
        grade = identity_grade(g["method_id"].iloc[0], did, mid)  # dataset/metric-level grade

        def cluster_key(e):
            sp = _split_family(e.split)
            ms = _norm(e.metric)   # raw metric surface (variant signal within canonical metric)
            un = _norm(e.unit)
            if not sp or not ms or not un:
                return None  # comparability-unknown (missing a facet)
            return (sp, ms, un)

        clusters = defaultdict(list)
        unknown = []
        for e in entries:
            k = cluster_key(e)
            (unknown if k is None else clusters[k]).append(e)

        def better(a, b):  # is a strictly better than b under direction
            if direction == "higher":
                return a > b
            if direction == "lower":
                return a < b
            return None

        # naive winner (best value overall), direction-aware
        if direction == "higher":
            naive_win = max(entries, key=lambda e: e.value)
        elif direction == "lower":
            naive_win = min(entries, key=lambda e: e.value)
        else:
            naive_win = None

        principal = max(clusters.values(), key=len) if clusters else []
        if principal and direction in ("higher", "lower"):
            cleaned_win = (max if direction == "higher" else min)(principal, key=lambda e: e.value)
        else:
            cleaned_win = None

        winner_changed = (naive_win is not None and cleaned_win is not None
                          and naive_win.method_id != cleaned_win.method_id)

        # pairwise classification, separating CONFIRMED-incomparable (both protocols known and
        # different) from UNKNOWN (at least one entry missing a facet -> cannot confirm).
        cl = {}                       # entry -> cluster index (known) or None (unknown)
        for ci, es in enumerate(clusters.values()):
            for e in es:
                cl[id(e)] = ci
        for e in unknown:
            cl[id(e)] = None
        pairs = list(combinations(entries, 2))
        n_pairs = len(pairs) or 1
        comparable = sum(1 for a, b in pairs if cl[id(a)] is not None and cl[id(a)] == cl[id(b)])
        confirmed_incomp = sum(1 for a, b in pairs if cl[id(a)] is not None and cl[id(b)] is not None
                               and cl[id(a)] != cl[id(b)])
        unknown_pairs = n_pairs - comparable - confirmed_incomp
        pair_comparable_frac = comparable / n_pairs
        pair_confirmed_incomp_frac = confirmed_incomp / n_pairs
        pair_unknown_frac = unknown_pairs / n_pairs

        # top-3 contamination: of the naive top-3, classify in-principal / confirmed-other / unknown
        if direction in ("higher", "lower"):
            ordered = sorted(entries, key=lambda e: e.value, reverse=(direction == "higher"))
            top3 = ordered[:3]
            principal_ids = {id(e) for e in principal}
            t3_confirmed_other = sum(1 for e in top3 if cl[id(e)] is not None and id(e) not in principal_ids) / len(top3)
            t3_unknown = sum(1 for e in top3 if cl[id(e)] is None) / len(top3)
            top3_contam = round(t3_confirmed_other + t3_unknown, 4)  # not confirmed comparable to principal
        else:
            top3_contam = t3_confirmed_other = t3_unknown = None
        principal_coverage = len(principal) / len(entries)

        lb_rec = {
            "leaderboard_id": f"{did}|{mid}", "dataset": ds_name, "metric": metric_name,
            "metric_direction": direction, "identity_grade": grade,
            "n_entries": len(entries), "n_methods": len(methods),
            "n_comparable_clusters": len(clusters), "n_comparability_unknown": len(unknown),
            "principal_cluster_size": len(principal),
            "naive_winner": {"method": cname(naive_win.method_id, naive_win.method),
                             "value": round(float(naive_win.value), 4),
                             "paper_id": naive_win.paper_id} if naive_win is not None else None,
            "cleaned_principal_winner": {"method": cname(cleaned_win.method_id, cleaned_win.method),
                                         "value": round(float(cleaned_win.value), 4),
                                         "paper_id": cleaned_win.paper_id,
                                         "cluster": list(max(clusters, key=lambda k: len(clusters[k])))}
            if cleaned_win is not None else None,
            "winner_changed": bool(winner_changed),
            "principal_cluster_coverage": round(principal_coverage, 4),
            "pair_comparable_fraction": round(pair_comparable_frac, 4),
            "pair_confirmed_incomparable_fraction": round(pair_confirmed_incomp_frac, 4),
            "pair_unknown_fraction": round(pair_unknown_frac, 4),
            "top3_not_confirmed_comparable_to_principal": (round(top3_contam, 4) if top3_contam is not None else None),
            "clusters": [
                {"protocol": {"split": k[0], "metric_surface": k[1], "unit": k[2]},
                 "ranking": [{"method": cname(e.method_id, e.method), "value": round(float(e.value), 4),
                              "arxiv_id": e.paper_id,
                              "arxiv_abs_url": f"https://arxiv.org/abs/{e.paper_id}",
                              "source_location": str(e.source_block)}  # section/table label; NO excerpt text
                             for e in sorted(es, key=lambda e: e.value, reverse=(direction == "higher"))]}
                for k, es in sorted(clusters.items(), key=lambda kv: -len(kv[1]))],
            "comparability_unknown_entries": [
                {"method": cname(e.method_id, e.method), "value": round(float(e.value), 4),
                 "paper_id": e.paper_id, "reason": "missing split/metric-surface/unit facet"}
                for e in unknown],
        }
        leaderboards.append(lb_rec)

        agg["n_leaderboards"] += 1
        agg["sum_comparable"] += pair_comparable_frac
        agg["sum_confirmed_incomp"] += pair_confirmed_incomp_frac
        agg["sum_unknown"] += pair_unknown_frac
        if len(clusters) >= 2:
            agg["n_multicluster"] += 1
        agg["sum_principal_cov"] += principal_coverage
        agg["by_grade"][grade]["n"] += 1
        if direction in ("higher", "lower"):
            agg["known_direction"] += 1
            if winner_changed:
                agg["winner_changed"] += 1
                agg["by_grade"][grade]["winner_changed"] += 1
            if top3_contam is not None:
                agg["sum_top3_contam"] += top3_contam
                agg["n_top3"] += 1

    with open(OUT / "cleaned_leaderboards.jsonl", "w") as f:
        for lb in leaderboards:
            f.write(json.dumps(lb, ensure_ascii=False) + "\n")

    summary = {
        "n_input_tuples": n_all, "quarantined_unsupported": quarantined,
        "n_leaderboards_2plus_methods": agg["n_leaderboards"],
        "n_with_known_direction": agg["known_direction"],
        "n_leaderboards_with_demonstrable_multiprotocol": agg["n_multicluster"],
        "mean_principal_cluster_coverage": round(agg["sum_principal_cov"] / agg["n_leaderboards"], 4) if agg["n_leaderboards"] else None,
        "DIVERGENCE": {
            "top1_winner_change_rate": round(agg["winner_changed"] / agg["known_direction"], 4) if agg["known_direction"] else None,
            "top1_winner_change_note": "naive best-value winner differs from the dominant-protocol (principal-cluster) winner",
            "mean_top3_not_confirmed_comparable": round(agg["sum_top3_contam"] / agg["n_top3"], 4) if agg["n_top3"] else None,
            "mean_pair_comparable_fraction": round(agg["sum_comparable"] / agg["n_leaderboards"], 4) if agg["n_leaderboards"] else None,
            "mean_pair_CONFIRMED_incomparable_fraction": round(agg["sum_confirmed_incomp"] / agg["n_leaderboards"], 4) if agg["n_leaderboards"] else None,
            "mean_pair_unknown_fraction": round(agg["sum_unknown"] / agg["n_leaderboards"], 4) if agg["n_leaderboards"] else None,
        },
        "by_identity_grade": {gr: {"n_leaderboards": v["n"],
                                   "winner_change_rate": round(v["winner_changed"] / v["n"], 4) if v["n"] else None}
                              for gr, v in agg["by_grade"].items()},
        "kendall_tau_note": "Within a comparable cluster the cleaned ranking is the value "
                            "ranking, identical to the naive order restricted to that cluster, "
                            "so rank correlation among comparable entries is 1.0 by "
                            "construction. The divergence is STRUCTURAL (invalid cross-protocol "
                            "comparisons), captured by the contamination metrics above, not by "
                            "within-cluster reordering.",
    }
    (OUT / "divergence_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT/'cleaned_leaderboards.jsonl'} ({len(leaderboards)} leaderboards)")


if __name__ == "__main__":
    main()

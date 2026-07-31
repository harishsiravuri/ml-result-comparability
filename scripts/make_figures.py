"""Manuscript figures for the cleaned-leaderboards finding (deterministic, $0).
House style: Okabe-Ito colorblind-safe palette, single-column width, serif, vector PDF + PNG.
Reads existing outputs only (divergence_by_grain.json, cleaned_leaderboards.jsonl).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# Embed scalable (TrueType/Type 42) fonts, not Type 3 bitmaps, per ACM proceedings requirements.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.paths import REPO_ROOT  # noqa: E402

D = REPO_ROOT / "data" / "cleaned_leaderboards"
FIG = REPO_ROOT / "figures"
FIG.mkdir(exist_ok=True)

# Okabe-Ito palette
OI = {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
      "grey": "#999999"}
SPLIT_COLOR = {"test": OI["blue"], "val": OI["orange"], "full": OI["green"],
               "trainval": OI["purple"], "train": OI["vermillion"], "unknown": OI["grey"]}
# marker shape as a redundant channel to color, for grayscale and color-vision robustness
SPLIT_MARKER = {"test": "o", "val": "s", "full": "^", "trainval": "D", "train": "v", "unknown": "x"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL = 3.4  # single-column inches


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def fig1_threeway(grains):
    order = ["all_leaderboards (conservative whole-corpus)", "all_pwc (PRIMARY, well-specified)",
             "demonstrably_multiprotocol (>=2 comparable clusters)"]
    labels = ["all\n(whole-corpus)", "all_pwc\n(primary)", "multi-protocol\n(n=215)"]
    g = {x["grain"]: x for x in grains}
    comp = [g[k]["mean_pair_comparable"] for k in order]
    inc = [g[k]["mean_pair_confirmed_incomparable"] for k in order]
    unk = [g[k]["mean_pair_unknown"] for k in order]
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    y = range(len(order))
    ax.barh(y, comp, color=OI["green"], label="comparable")
    ax.barh(y, inc, left=comp, color=OI["vermillion"], label="confirmed incomparable")
    ax.barh(y, unk, left=[c + i for c, i in zip(comp, inc)], color=OI["grey"], label="unknown")
    ax.set_yticks(list(y)); ax.set_yticklabels(labels)
    ax.set_xlabel("mean fraction of head-to-head pairs"); ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, columnspacing=1.0)
    save(fig, "fig1_threeway_pair_breakdown")


def fig2_winner_change(grains):
    order = ["all_leaderboards (conservative whole-corpus)", "all_pwc (PRIMARY, well-specified)",
             "partial_pwc", "hash_only", "demonstrably_multiprotocol (>=2 comparable clusters)"]
    # reader-facing names; the middle three are identity grades, "all" and "multi-protocol" are not
    labels = ["all", "fully canonical", "partially canonical", "hash-only", "multi-protocol"]
    g = {x["grain"]: x for x in grains}
    vals = [g[k]["winner_change_rate"] for k in order]
    ns = [g[k]["n_leaderboards"] for k in order]
    ci = [g[k]["winner_change_ci95"] for k in order]
    lo = [v - c[0] for v, c in zip(vals, ci)]; hi = [c[1] - v for v, c in zip(vals, ci)]
    xpos = [0.0, 1.5, 2.6, 3.7, 5.2]  # gaps set the identity-grade trio apart from "all" and "multi-protocol"
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    ax.bar(xpos, vals, width=0.85, yerr=[lo, hi], color=OI["blue"], capsize=2, error_kw={"lw": 0.8})
    ax.set_xticks(xpos); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=6.6)
    ax.set_ylabel("naive top-1 winner-change rate"); ax.set_ylim(0, 0.64); ax.set_xlim(-0.6, 5.9)
    ax.set_xlabel("leaderboard subset")
    for i, (xi, v, nn) in enumerate(zip(xpos, vals, ns)):
        ax.text(xi, v + hi[i] + 0.012, f"{v*100:.0f}%\nn={nn}", ha="center", va="bottom", fontsize=5.6)
    # dotted dividers separate the three identity grades from the "all" and "multi-protocol" subsets;
    # the grouping is explained in the caption rather than a floating title
    ax.axvline(0.75, color=OI["grey"], lw=0.5, ls=":")
    ax.axvline(4.45, color=OI["grey"], lw=0.5, ls=":")
    save(fig, "fig2_winner_change_by_grade")


def _split_of(cluster_protocol):
    return cluster_protocol["split"] if cluster_protocol else "unknown"


def fig3_examples(lbs):
    def get(dsname, met):
        for l in lbs:
            if l["dataset"] == dsname and l["metric"].lower() == met.lower():
                return l
        return None

    def dedup_top(lb, k=8):
        # Collapse near-duplicate method variants to the best value per (base method, split),
        # where the base name is the label up to the first parenthesised descriptor. A method
        # that appears under two splits keeps both markers, disambiguated by a split suffix.
        best = {}
        for c in lb["clusters"]:
            sp = _split_of(c["protocol"])
            for e in c["ranking"]:
                base = e["method"].split(" (")[0].strip()
                key = (base, sp)
                if key not in best or e["value"] > best[key][0]:
                    best[key] = (e["value"], base, sp)
        pts = sorted(best.values(), key=lambda t: -t[0])[:k]
        seen = [b for _, b, _ in pts]
        dup = {b for b in seen if seen.count(b) > 1}
        pts = [(v, (f"{b} ({sp})" if b in dup else b), sp) for v, b, sp in pts]
        return pts

    # left: a split-confounded leaderboard (val entries outrank test); right: a clean single-split leaderboard
    ex = [("PASCAL VOC 2012", "miou"), ("Kinetics-400", "top-1")]
    fig, axes = plt.subplots(1, 2, figsize=(COL * 2.05, 2.4))
    for ax, (ds, met) in zip(axes, ex):
        lb = get(ds, met)
        pts = dedup_top(lb)
        naive_val = max(p[0] for p in pts)
        ys = range(len(pts))
        for yi, (v, m, sp) in zip(ys, pts):
            ax.scatter([v], [yi], color=SPLIT_COLOR.get(sp, OI["grey"]),
                       marker=SPLIT_MARKER.get(sp, "o"), s=26, zorder=3)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([m for v, m, sp in pts])
        ax.invert_yaxis()
        ax.set_xlabel(f"{lb['metric']} (higher better)")
        ax.set_title(lb["dataset"], fontsize=7.5)
        ax.axvline(naive_val, color=OI["grey"], lw=0.6, ls=":", zorder=1)
    # shared legend: color encodes the split, which is the information the figure conveys
    handles = [plt.Line2D([0], [0], marker=SPLIT_MARKER.get(s, "o"), ls="", color=SPLIT_COLOR[s], label=s)
               for s in ("test", "val")]
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02), title="observed-protocol cluster (split)")
    save(fig, "fig3_worked_examples_naive_vs_cleaned")


def fig4_census(ra):
    # Uses the cell-clustered bootstrap intervals so the figure matches Table 1 exactly.
    c = ra["item1_2_census_cluster_aware"]

    def row(key):
        cc = c[key]["cell_cluster_[lo,hi]"]  # [point, lo, hi]
        return cc[0], cc[1], cc[2]

    # Row labels carry the category, so hue adds nothing; a single neutral color is used for the
    # three mutually exclusive prevalence rows, and a lighter tint marks the two decomposition rows.
    PART = OI["blue"]; SUB = "#A9CCE3"  # lighter tint of the same blue
    rows = [
        ("real disagreement", row("real_disagreement"), PART),
        ("within noise", row("within_noise"), PART),
        ("extraction / identity artifact", row("extraction_or_identity_artifact"), PART),
        ("  -> protocol artifact", row("protocol_artifact"), SUB),
        ("  -> genuine conflict", row("genuine_conflict"), SUB),
    ]
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    ys = list(range(len(rows)))[::-1]
    for y, (lab, ci, col) in zip(ys, rows):
        pt, lo, hi = ci
        ax.barh(y, pt, color=col, height=0.62)
        ax.errorbar(pt, y, xerr=[[pt - lo], [hi - pt]], fmt="none", ecolor=OI["black"], lw=0.8, capsize=2)
        ax.text(hi + 0.01, y, f"{pt:.2f}", va="center", fontsize=6.3)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("share of candidate disagreement pairs (95% CI)"); ax.set_xlim(0, 0.62)
    ax.axhline(1.5, color=OI["grey"], lw=0.5, ls=":")
    save(fig, "fig4_census_prevalence")


def fig5_scale_cost(sc):
    labels = ["our pipeline\n(whole census)", "frontier-only\n(all candidates)",
              "frontier-only\n(field leaderboards)"]
    vals = [sc["our_actual_total_pipeline_spend_usd"],
            sc["projected_frontier_only_judge_all_candidates"]["cost_usd"],
            sc["projected_frontier_only_field_leaderboard_cleaning"]["cost_usd"]]
    colors = [OI["green"], OI["skyblue"], OI["vermillion"]]
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    x = range(len(vals))
    ax.bar(x, vals, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("USD (log scale)")
    for xi, v in zip(x, vals):
        ax.text(xi, v * 1.15, f"${v:,.0f}" if v >= 100 else f"${v:.2f}", ha="center", va="bottom", fontsize=6.4)
    ax.set_ylim(1, 20000)
    save(fig, "fig5_scale_cost")


def main():
    grains = json.load(open(D / "divergence_by_grain.json"))["grains"]
    lbs = [json.loads(l) for l in open(D / "cleaned_leaderboards.jsonl") if l.strip()]
    ra = json.load(open(REPO_ROOT / "data" / "census" / "revision_analysis.json"))
    sc = json.load(open(REPO_ROOT / "data" / "census" / "scale_cost.json"))
    fig1_threeway(grains)
    fig2_winner_change(grains)
    fig3_examples(lbs)
    fig4_census(ra)
    fig5_scale_cost(sc)
    print("wrote figures:", sorted(p.name for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()

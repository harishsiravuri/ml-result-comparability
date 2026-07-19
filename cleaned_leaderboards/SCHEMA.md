# Schema: comparability-partitioned leaderboards

**Terminology (important).** A "comparable cluster" and the `..._comparable...` fields below
denote a **same-observed-protocol** group: the entries show no visible difference on the
represented protocol facets (split family, metric surface, unit). This is **not** confirmed
comparability. Human validation (see the paper, Table 4) finds that entries sharing an
observed protocol are still frequently incomparable on hidden facets (only 2 of 15 audited
same-protocol pairs were judged comparable). The reliable signal is the reverse: entries in
**different** clusters are incomparable (12 of 12 audited). Read every `..._comparable...`
field as `..._same_observed_protocol...`.

`cleaned_leaderboards.jsonl` — one JSON object per canonical (dataset, metric) leaderboard
with at least two methods. Built deterministically from the frozen tuple index; see
`checkpoints/cleaned_leaderboards_report.md` for the policy and the divergence finding.

Per-leaderboard fields:
- `leaderboard_id`: "<dataset_id>|<metric_id>" (canonical identifiers).
- `dataset`, `metric`, `metric_direction`: canonical names and higher|lower|unknown.
- `identity_grade`: all_pwc | partial_pwc | hash_only (the canonicalization confidence;
  all_pwc is cleanest).
- `n_entries`, `n_methods`, `n_comparable_clusters` (same-observed-protocol clusters), `n_comparability_unknown`,
  `principal_cluster_size`, `principal_cluster_coverage`.
- `naive_winner`: {method, value, paper_id} — the best value ignoring protocol.
- `cleaned_principal_winner`: {method, value, paper_id, cluster} — the best value within the
  dominant same-observed-protocol cluster.
- `winner_changed`: whether the naive winner differs from the dominant-protocol winner.
- `pair_comparable_fraction` (share of head-to-head pairs sharing the same observed protocol; NOT confirmed comparable), `pair_confirmed_incomparable_fraction`, `pair_unknown_fraction`:
  head-to-head pair classification (sum to 1). CONFIRMED incomparable means both entries have
  known, differing protocol facets; UNKNOWN means at least one entry is missing a facet.
- `top3_not_confirmed_comparable_to_principal`: fraction of the naive top-3 not sharing the
  dominant cluster's observed protocol.
- `clusters`: list, largest first. Each: {protocol: {split, metric_surface, unit},
  ranking: [{method, value, arxiv_id, arxiv_abs_url, source_location}]} ranked by direction.
- `comparability_unknown_entries`: entries excluded from clustering (missing a facet), with
  the reason. These are FLAGGED, not forced into a cluster.

Provenance: every entry carries a POINTER only (arxiv_id, arxiv_abs_url, and the
source_location = a section or table label). No excerpt text is redistributed; consult the
cited arXiv paper directly. The curated Papers-with-Code value never enters the cleaning (the
cleaning uses the papers' own reported values and extracted protocol facets only).

`divergence_summary.json` — the field-wide naive-versus-cleaned divergence aggregates.

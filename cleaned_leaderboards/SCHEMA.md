# Schema: comparability-cleaned leaderboards

`cleaned_leaderboards.jsonl` — one JSON object per canonical (dataset, metric) leaderboard
with at least two methods. Built deterministically from the frozen tuple index; see
`checkpoints/cleaned_leaderboards_report.md` for the policy and the divergence finding.

Per-leaderboard fields:
- `leaderboard_id`: "<dataset_id>|<metric_id>" (canonical identifiers).
- `dataset`, `metric`, `metric_direction`: canonical names and higher|lower|unknown.
- `identity_grade`: all_pwc | partial_pwc | hash_only (the canonicalization confidence;
  all_pwc is cleanest).
- `n_entries`, `n_methods`, `n_comparable_clusters`, `n_comparability_unknown`,
  `principal_cluster_size`, `principal_cluster_coverage`.
- `naive_winner`: {method, value, paper_id} — the best value ignoring protocol.
- `cleaned_principal_winner`: {method, value, paper_id, cluster} — the best value within the
  dominant comparable cluster.
- `winner_changed`: whether the naive winner differs from the dominant-protocol winner.
- `pair_comparable_fraction`, `pair_confirmed_incomparable_fraction`, `pair_unknown_fraction`:
  head-to-head pair classification (sum to 1). CONFIRMED incomparable means both entries have
  known, differing protocol facets; UNKNOWN means at least one entry is missing a facet.
- `top3_not_confirmed_comparable_to_principal`: fraction of the naive top-3 not confirmed
  comparable to the dominant cluster.
- `clusters`: list, largest first. Each: {protocol: {split, metric_surface, unit},
  ranking: [{method, value, paper_id, evidence_quote, source_block}]} ranked by direction.
- `comparability_unknown_entries`: entries excluded from clustering (missing a facet), with
  the reason. These are FLAGGED, not forced into a cluster.

Provenance: every entry carries paper_id, the evidence quote, and the source block. The
curated Papers-with-Code value never enters the cleaning (the cleaning uses the papers' own
reported values and extracted protocol facets only).

`divergence_summary.json` — the field-wide naive-versus-cleaned divergence aggregates.

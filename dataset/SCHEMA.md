# Schema: candidate disagreement dataset

`comparekg_candidates.jsonl` -- one JSON object per candidate cross-paper disagreement pair
(3,058 rows). `comparekg_gold.jsonl` -- the 200-pair human-labeled reference set (same fields
plus populated `human_label`).

Per-row fields:
- `pair_id`: stable identifier.
- `cell`: {`method`, `dataset`, `metric`, `metric_direction`} -- the canonical (method, dataset, metric) cell.
- `left`, `right`: the two sides, each a POINTER only: {`arxiv_id`, `arxiv_abs_url`, `arxiv_version` (null; not recorded in the snapshot), `source_location` (section or table label), `value`, `unit`}. No excerpt text is redistributed.
- `identity_grade`: `all_pwc` (fully canonical), `partial_pwc` (partially canonical), or `hash_only` (hash-only).
- `noise_decision`: {`beyond_noise` (bool), `gap`, `threshold`, `dispersion_source` (`reported` | `default`)} -- the default-tolerance exceedance decision (see the paper; not a significance test).
- `value_gap`, `rel_gap`, `unit_scale_reconciled`: magnitude fields.
- Label provenance (one row may carry none, one, or both):
  - `human_label` (or null), `human_validated` (bool) -- populated only in the reference set (200 pairs).
  - `model_suggested_label` (or null), `model_confidence` (or null) -- populated on 523 candidates; null elsewhere. NO field-wide judge pass was run.
  - `label_source`: `human` | `model` | null.
- `judge_frozen`: the cached judge trace on the 523 judged rows: {`backbones`, `cause`, `rule_label`, `top_level`, `confidence`, `rationale`}; null elsewhere.
- `n_protocols_on_dataset_metric`, `pair_type`, `task_family`, `auto_derivable_crosscheck`: context fields.

Label coverage: 200 human (reference set) and 523 model-suggested candidates, overlapping on 184,
so 539 of the 3,058 carry at least one label and 2,519 carry none. Model-suggested labels are not gold.

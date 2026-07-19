# Dataset card: comparekg — cross-paper result-cell disagreement census

## What this is

A provenance-grounded dataset of candidate cross-paper QUANTITATIVE result-cell
disagreements mined from a frozen field-scale leaderboard archive, with a
human-validated subset attributing each disagreement to a cause. The unit is a PAIR: two
papers reporting differing values for what a canonicalization matched as the same
(method, dataset, metric) result cell.

Two files (`data/release/`):
- `comparekg_gold.jsonl` (200 rows): the HUMAN-VALIDATED subset. Each row carries full
  provenance, the deterministic noise decision, the frozen-judge prediction (where the
  judge ran), the AUTHOR label, the auto-derivable cross-check, the dev/test split, and
  the reliability flags. This is the headline released artifact.
- `comparekg_candidates.jsonl` (3,058 rows): the full candidate resource. Cell identity,
  source papers and spans, reported values, and the deterministic noise decision. The
  frozen-judge cause is present only where the judge ran (test-gold plus dev beyond-noise,
  523 rows); elsewhere it is null and flagged. No field-wide judge pass was run.

## Exact definition

A CANDIDATE cross-paper result-cell inconsistency is an unordered pair of result-cell
tuples from two DIFFERENT papers that share the same canonical (method_id, dataset_id,
metric_id) identity and report differing values (after a percent-versus-fraction unit
reconciliation). The author labels each sampled pair with one of seven labels:
`split`, `metric_variant`, `evaluation_setting`, `citation_reporting_discrepancy`,
`genuine_conflict` (the five "real disagreement" causes), and `within_noise`,
`extraction_artifact` (the two "not a real disagreement" labels).

## Provenance and the frozen-snapshot caveat

The structured layer derives from the frozen Papers-with-Code archive. Papers with Code
was shut down by Meta on 2025-07-24; the surviving data is the Hugging Face
`pwc-archive` mirror, frozen 2025-07-28, parquet format, CC-BY-SA 4.0. All eleven raw
parquet files were re-verified by sha256 on 2026-06-18 (see `data/SNAPSHOT.md`). For a
frozen census this static field state is a strength: stable, citable, reproducible. The
result-cell tuples are Paper 2's GPT-4o-mini extraction with verification metadata; cell
identity uses the reused canonicalization resolver.

## Splits

Frozen dataset-level dev/test split (seed 13, dev fraction 0.30) so all protocol variants
of a dataset stay on one side. Candidates: 689 dev / 2,369 test pairs; 0 datasets
straddle. The human-gold 200: 42 dev / 158 test. The single-shot validation used the 158
test-gold pairs.

## Annotation protocol (read with the limitations)

- Single annotator (the author), labeling blind to the model predictions, the noise
  decision, and the Papers-with-Code curated value. The instrument is
  `docs/annotation_guide.md`.
- Stratified sample of 200 pairs (seed 20260618) across pair type, identity grade,
  value-gap tercile, and a deterministic cause proxy, with a floor on the rule-fired
  strata so the split and metric_variant sub-types are represented.
- INTRA-ANNOTATOR test-retest reliability (45 pairs, re-labeled after a gap, presented in
  a shuffled order): Cohen kappa 0.734 for the inconsistency decision and 0.676 for the
  seven-way cause. The first pass treated all 200 pairs identically (no retest marker), so
  the reliability is not optimistically biased.
- A SECOND annotator was NOT used. A 50-pair second-annotator overlap is flagged
  (`in_second_annotator`) so a second Cohen kappa can be added later without rework. This
  is a stated limitation.
- Auto-derivable cross-check (QA, not a target): author labels versus the deterministic
  rule on rule-fired pairs (split rule agreement 0.35, metric_variant rule 0.00; the
  metric_variant rule does not match human judgment).

## The census (human-validated, post-stratified, detector-independent)

Reweighting the stratified gold to the 3,058-candidate population (stratified bootstrap
95% intervals):
- real cross-paper disagreement: 0.450 [0.392, 0.511]
- extraction or identity-match artifact: 0.270 [0.223, 0.322]
- within noise: 0.248 [0.211, 0.285]
- protocol_artifact (any sub-type) 0.380 [0.320, 0.442] versus genuine_conflict 0.063
  [0.032, 0.096]: about 84 percent of REAL disagreements are protocol-induced
  incomparability, not genuine conflicts.
- leaderboard audit: of human-confirmed real disagreements on a dataset+metric hosting
  more than one Papers-with-Code leaderboard, 0.885 are protocol artifacts (incomparable).

COVERAGE caveat (important). The extraction or identity-match artifact rate concentrates
in hash-matched identities. On the fully canonical (all_pwc) cells the artifact rate is
only 0.073 and the real-disagreement rate is 0.634; on partial_pwc cells the artifact rate
is 0.348. Restricting to all_pwc cells gives a much cleaner census at the cost of coverage
(about 28 percent of real disagreements retained). The artifact rate is therefore a
candidate-SURFACING (canonicalization) limitation. A sensitivity using the Paper 2.2
trust-scored knowledge graph confirms its per-fact trust does NOT separate identity
artifacts (area under the curve 0.511); the lever is the canonicalization confidence, not
value trust (`data/census/paper2_2_sensitivity.json`).

## Honest method characterization (the detector did NOT beat a bare frontier model)

The preregistered kill-gate TRIPPED and we executed fallback (i) (see
`FAILURE_ANALYSIS.md`). On the 158 test-gold pairs:
- the inconsistency DECISION matches the author (F1 0.742, at or above the 0.70 floor) and
  beats the naive-rule and textual-NLI baselines, but it does NOT beat the strongest bare
  frontier model (Opus 4.8, 0.760); they are a per-pair statistical tie;
- the top-level 4-class cause macro F1 is 0.417 (a clean miss of the 0.55 bar); the frozen
  split-and-metric_variant macro F1 is 0.522 (a close miss of the 0.60 bar; an LLM-only
  ablation reaches 0.635);
- the rule-anchored judge (frozen) under-performs a plain LLM judge on the validated core;
  the metric_variant rule has zero precision and recall against gold.

The contribution is therefore the human-validated CENSUS, the validated inconsistency
DECISION, this released DATASET, and the SCALE argument (the structured layer surfaces and
judges field-wide candidates that a bare frontier model cannot), NOT a detector that beats
frontier models. Per-pair, a bare frontier model is competitive. The released judge
predictions are the frozen rule-first judge; treat them as a weak attribution baseline,
not a validated label.

## Schema (per row)

`pair_id, split, cell{method_id, dataset_id, metric_id, method, dataset, metric,
metric_direction}, identity_grade, pair_type, task_family, n_protocols_on_dataset_metric,
value_gap, rel_gap, unit_scale_reconciled, left{...}, right{...}, noise_decision{beyond_noise,
range_type, gap, threshold, dispersion_source}, judge_frozen{cause, top_level, rationale,
rule_label, confidence, backbones} | null, auto_derivable_crosscheck`. LABEL PROVENANCE on
every row: `human_label` (or null), `model_suggested_label` (or null), `label_source`
(human | model | null), `model_confidence` (or null), `human_validated` (bool). Gold rows
add `author_label, author_confidence_1to5, in_test_retest, in_second_annotator`. Each side is a POINTER only:
`arxiv_id, arxiv_abs_url, arxiv_version (null; not recorded in the snapshot), value, unit,
split, is_own_result, source_location (a section or table label), quote_verified,
self_consistency, critic_verdict`. NO evidence-span excerpt text is included.

## Licensing and intended use

Method (how the rows were produced): result tuples (method, dataset, metric, value) and
their locations (a section or table label) were extracted from the FULL TEXT of the arXiv
papers linked in the frozen Papers-with-Code snapshot. The public release does NOT
redistribute any evidence text: each row carries a POINTER only (arXiv identifier, arXiv
abstract URL, and the source location) plus the reported value. To read a value in context,
consult the cited arXiv paper directly.

Licensing: because the release contains no third-party text, the ENTIRE release (this
dataset and the comparability-cleaned leaderboards) is licensed under CC-BY-SA 4.0 (full
statement in `LICENSES.md`); the Papers-with-Code-derived facts inherit CC-BY-SA 4.0 from
the frozen archive. The repository code is MIT licensed.

Intended use: studying cross-paper quantitative comparability and incomparability in a
field's literature, and as a benchmark for disagreement detection and protocol-cause
attribution. NOT intended to adjudicate which paper is "correct"; the curated leaderboard
value is deliberately excluded from the attribution inputs.

## Comparability-cleaned leaderboards (released resource, Phase 6 addition)

`data/cleaned_leaderboards/` releases, for each canonical (dataset, metric) leaderboard
with at least two methods, a conservatively cleaned ranking with full per-entry provenance
and comparability flags (`SCHEMA.md`, `MANIFEST.json`,
`checkpoints/cleaned_leaderboards_report.md`).

How "comparable" is defined: two entries are treated as comparable only when their explicit,
extractable protocol facets (normalized split family, raw metric surface, unit) are all known
and equal. Entries missing any facet are flagged `comparability-unknown` and are NOT grouped
with others (we prefer flagging over false grouping). Extraction artifacts (critic verdict
UNSUPPORTED) are quarantined. Within a comparable cluster, entries are ranked by the metric
direction; cross-cluster comparisons are marked not directly comparable.

Coverage and finding (descriptive, no model bar). The honest headline is the three-way pair
split: field-wide, of head-to-head pairs, 0.386 are comparable, only 0.015 are CONFIRMED
incomparable, and 0.599 are comparability-UNKNOWN (the extracted metadata is too sparse to
adjudicate most comparisons). Winner-change by grain (the naive best-value winner differs from
the dominant-protocol winner; Wilson 95 percent intervals): the PRIMARY all_pwc (well-specified)
cut 0.188 [0.089, 0.353] (n=32); the 215 demonstrably multi-protocol leaderboards 0.463 [0.394,
0.534] (confirmed-incomparable pairs reach 0.314 there); the conservative whole-corpus rate
0.080 [0.071, 0.091] (a lower bound, since most leaderboards lack the metadata to detect the
problem). Canonical failure mode: a validation-split result ranked above test-split results
(PASCAL VOC 2012 mIoU, naive winner 94.5 on val versus the test-cluster winner 89.8; Cityscapes
mIoU, naive 87.4 on val versus test winner 82.8).

Reconciliation (oracle versus realizable). The human-adjudicated census (conditioned on a
confirmed real disagreement) finds about 84 percent of real disagreements incomparable; the
field-wide automated cleaning can confirm comparability for only about 39 percent of pairs
(about 60 percent unknown) because protocol metadata is not reliably extractable. These are
complementary: incomparability is the norm where a human can adjudicate, but at scale it is
mostly unverifiable. This quantifies, at field scale, the old Chapter 3 protocol-unextractability
finding (about 73 percent of fine protocol axes not reliably extractable from paper text).

Validation is inherited from Phase 5 (decision F1 0.742; about 84 percent of real disagreements
are protocol artifacts); this resource introduces NO new preregistered bar and does NOT re-grade
the spent detector test. The cleaning uses deterministic protocol facets and therefore inherits
the Phase 5 facet-rule limitations, so the confirmed-incomparable fraction is a conservative
lower bound and the all_pwc view is cleanest.

Intended use: a downstream assistant can answer ranking questions from these cleaned
leaderboards to keep comparisons fair by construction (rank within a comparable cluster,
surface each cluster's protocol, and flag or refuse cross-cluster and comparability-unknown
comparisons rather than assert a single winner). Treat the clusters as conservative and
provenance-grounded, not as ground truth.

## Reproduction

Build candidates (`scripts/build_candidates.py`), the noise model
(`scripts/run_noise_model.py`), the judge (`scripts/run_judge_dev.py`,
`scripts/run_phase5.py`), the baselines (`scripts/run_baselines.py`), the gold sample
(`scripts/draw_gold_sample.py`), scoring (`scripts/score_phase5.py`), the census
(`scripts/phase5_census.py`), and this release (`scripts/package_release.py`). The
preregistration is frozen at commit 06267c0.

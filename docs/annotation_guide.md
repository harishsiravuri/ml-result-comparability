# Annotation guide: cross-paper result-cell disagreement (comparekg, Phase 4)

This guide operationalizes the human-gold annotation for the Chapter 3 census. It is
relayed for strategic review BEFORE any blind labeling begins. The protocol is fixed by
the frozen preregistration (06267c0); this document only makes it concrete. House style
follows the project: no contractions, no em dashes, active "we".

Instrument-refinement note (2026-06-18, before any labeling): the within-noise guidance
below was refined to be scale-aware per strategic review. The annotation guide is a
Phase-4 instrument product, not content frozen at the preregistration commit, so this is
an instrument refinement and not a preregistration deviation.

## What you are labeling

You will see 200 candidate pairs (`data/census/gold_sample.jsonl`). Each pair is two
papers (A and B) reporting a numeric result for what our pipeline matched as the same
canonical (method, dataset, metric) cell, with differing values. For each pair you make
ONE judgment: what best explains the difference between the two reported numbers.

You record one LABEL per pair from the seven-way space below, a confidence (1 to 5), and
an optional short note, in `data/census/gold_annotation_sheet.csv`.

## Blind protocol (important)

- The packet shows ONLY: the canonical cell, each side's reported value, unit, stated
  split, whether it was the paper's own result or a cited number, the source span
  (evidence quote and source block), and the surrounding table caption / table snippet /
  setup paragraph where available.
- The packet deliberately does NOT show: the method's prediction, the noise model's
  decision, or any Papers-with-Code curated "official" value. Do NOT look these up.
  Judge only from what is in the packet. This blindness is what makes the gold an
  independent check on the method.

## The seven labels

First decide whether the two numbers are really in disagreement beyond expected noise.
Judge whether the gap is plausibly run-to-run or seed variance FOR THIS METRIC AND TASK.
Prefer the reported confidence interval or standard deviation when one is given: a gap
inside it is noise. When no variance is reported, use a rough anchor and translate it to
the metric's actual scale -- many metrics on a 0-to-100 scale vary by about 0.5 to 2
points across seeds; for a metric reported in [0,1] the equivalent band is about 0.005 to
0.02; for other scales judge proportionally. When unsure whether a gap is noise, label
your best call and lower your confidence.

1. `within_noise`: the gap is plausibly run-to-run or seed variance for this metric and
   task (or is inside the reported confidence interval or standard deviation). Not a real
   disagreement.
2. `extraction_artifact`: this is NOT a genuine cross-paper disagreement because the
   pairing or a value is wrong: an extraction error in one reported number, or a spurious
   identity match (the two cells are actually different methods, datasets, or metrics).

If the disagreement is real and beyond noise, attribute its CAUSE:

3. `split`: a different data split or fold of the same dataset (for example test versus
   validation, a different fold, test-dev versus test, minival versus val).
4. `metric_variant`: the SAME metric computed differently. This INCLUDES a different
   cutoff (Recall@20 versus Recall@50, top-1 versus top-5, Hits@1 versus Hits@10) and a
   different aggregation (micro versus macro versus weighted, per-class versus overall,
   instance versus frame versus video), and filtered versus raw, constrained versus
   unconstrained. The metric NAME is the same family; only HOW it is cut off or averaged
   differs.
5. `evaluation_setting`: a different EXPERIMENTAL setting that is neither a split nor a
   metric cutoff/averaging. This INCLUDES few-shot k versus full or zero-shot, a
   different input resolution or image size, extra or different training or pre-training
   data, a different subtask or sub-benchmark, or a different inference budget.
6. `citation_reporting_discrepancy`: the two numbers are meant to refer to the SAME
   result, but one was mis-copied, mis-cited, or taken from a different source cell (for
   example B cites A but reports a slightly different number than A's own paper).
7. `genuine_conflict`: same protocol, independently produced, and the numbers still
   differ beyond noise (a real irreproducibility, not explained by any protocol cause).

Boundary rule (apply consistently): if the only difference is the metric's cutoff (@k,
top-k) or its aggregation, label `metric_variant`, NOT `evaluation_setting`. Reserve
`evaluation_setting` for differences in the experimental conditions that produced the
numbers. If you find this boundary genuinely hard on a given pair, label your best call
and lower your confidence; the test-retest pass measures how fuzzy this boundary is.

## Worked examples

- A reports 88.0 labelled "micro-F1", B reports 84.0 labelled "macro-F1" -> `metric_variant`.
- A reports Recall@20 = 0.18, B reports Recall@50 = 0.31 -> `metric_variant`.
- A reports 79.1 on "val", B reports 77.8 on "test" -> `split`.
- A reports 76.0 "5-shot", B reports 81.0 "full fine-tuning" -> `evaluation_setting`.
- A (the method's own paper) reports 71.2, B cites A as 70.9 with no protocol difference
  -> `citation_reporting_discrepancy`.
- Two papers both re-run the method under the same stated protocol and report 64.0 and
  61.5 with no protocol difference and gap beyond noise -> `genuine_conflict`.
- The two "matched" rows are actually different datasets (one is the multilingual split)
  -> `extraction_artifact`.
- A reports 80.1, B reports 80.4, same protocol, within seed noise -> `within_noise`.

## How the labels are scored (for context only; does not change how you label)

The inconsistency DECISION is derived from your label: a real beyond-noise disagreement
is any label in {split, metric_variant, evaluation_setting,
citation_reporting_discrepancy, genuine_conflict}; `within_noise` and
`extraction_artifact` are "not a real disagreement". We report precision, recall, and F1
of the method against your labels, separately for the decision and for the cause, and
per cause. The validated core is the decision, the top-level protocol-artifact-versus-other
call, and the split and metric_variant sub-types.

Note on the decision: your `within_noise` judgment partly overlaps the preregistered noise
model, so the decision F1 is read with that overlap in mind. Your independent contribution
to the decision is strongest in catching `extraction_artifact` and identity-mismatch pairs,
which the noise model cannot detect.

## Reliability and the second annotator

- INTRA-ANNOTATOR TEST-RETEST: 45 pairs are flagged (`in_test_retest` = 1). After you
  finish the main pass, wait at least one week, then re-label those 45 in
  `gold_annotation_sheet_retest.csv` WITHOUT looking at your first labels. We report
  Cohen kappa between the two passes (for the decision and for the cause) as the
  reliability statistic, in place of inter-annotator agreement.
- SECOND-ANNOTATOR HOOK: 50 pairs are flagged (`in_second_annotator` = 1). A second
  annotator may later double-label exactly these so a second Cohen kappa can be slotted
  in by the writing session without rework. No action is needed now.
- A second annotator is NOT used for the primary gold. We state this as a declared
  limitation in the dataset card and the manuscript.

## Mechanics

- Fill `pair_id`, `label` (one of the seven exact strings), `confidence_1to5`
  (1 = guess, 5 = certain), and an optional `note`, in
  `data/census/gold_annotation_sheet.csv`. Do not reorder rows.
- Label every pair. If truly undecidable after reading the context, use your best single
  label and set confidence to 1, and explain in the note.

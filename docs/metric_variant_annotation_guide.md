# Annotation guide: metric-variant recovery (item 5)

Companion to `data/census/metric_variant_packet.md` and
`data/census/metric_variant_sheet.csv` (130 pairs).

## Why this sample exists

`metric_variant` is the weakest facet in the chapter. At Gate 4 the deterministic
metric-variant rule agreed with your labels on **0 of 17** pairs, and the split-plus-variant
bar was a clean miss (macro-F1 0.522). A deterministic audit of all 3,058 census pairs now
explains why:

| Metric surfaces on the two sides | pairs |
|---|---|
| identical | 2,740 |
| a variant stated on one side only (`overall accuracy` vs `accuracy`) | 165 |
| the same surface spelled differently (`acc` vs `accuracy`) | 150 |
| genuinely different variant signatures | **3** |

and all **11** pairs you labeled `metric_variant` in the frozen gold have **identical** metric
surfaces. The old rule was firing on spelling and on one-sided statements, and could not have
found the real cases. Nine of those 11 do carry a variant cue (`@k`, `filtered`/`raw`,
`micro`/`macro`) in the table caption, table snippet or setup paragraph.

So a metric variant is something you read out of the **context**, not out of the metric name.
This sample is built for that task. It is drawn fresh: none of the 11 frozen-gold pairs is in
it, and no cell from the frozen gold is in it.

## The question for each pair

These 130 pairs report **different** values on the same canonical cell. Ignore, for this task,
how large the difference is. The question is:

> Do the two numbers use the same metric **variant** — the same way of computing the metric —
> and if not, is the variant the reason they differ?

## Labels

| Label | Use when |
|---|---|
| `metric_variant` | The two sides compute the same metric name a different way, and that is the operative difference. Examples: micro vs macro averaging, filtered vs raw MRR, Hits@1 vs Hits@10, class-averaged vs instance-averaged, mAP at a different IoU threshold. |
| `same_metric_variant` | The evidence shows the two sides use the **same** metric variant, so whatever explains the value gap, it is not the metric variant. |
| `split` | The operative difference is the data split or fold. |
| `evaluation_setting` | The operative difference is some other protocol choice: few-shot vs full, extra training data, input resolution, a different subtask. |
| `within_noise` | The two values are close enough that run-to-run variation plausibly explains the gap. |
| `citation_reporting_discrepancy` | The numbers are meant to be the same result but one was mis-copied, mis-cited, or taken from a different cell. |
| `genuine_conflict` | Same protocol, independently produced, and they still differ beyond noise. |
| `extraction_artifact` | Not a genuine cross-paper pair: an extraction error, or a spurious identity match. |
| `undetermined` | The evidence shown does not let you decide. |

## Priority when more than one applies

    extraction_artifact  >  metric_variant  >  split  >  evaluation_setting
                         >  citation_reporting_discrepancy  >  genuine_conflict  >  within_noise

`metric_variant` sits above the other protocol causes **for this task only**, because the point
of the sample is to establish how often the variant is the operative difference and how often
it is recoverable at all. If a pair genuinely differs on both the split and the variant, label
`metric_variant` and say so in the note.

## The distinction that matters most

**`metric_variant` versus `evaluation_setting`.** A metric variant changes how the score is
**computed from the same predictions**: which averaging, which cutoff, which filtering.
An evaluation setting changes **what was run**: a different amount of supervision, different
inputs, a different subtask. Filtered vs raw MRR is a variant. Few-shot vs fully supervised is
a setting. If the difference would vanish by recomputing the metric on the same predictions, it
is a variant.

**`same_metric_variant` is a positive finding, not a fallback.** Use it when the context
actually tells you the variants match. If the context says nothing at all about the variant,
that is `undetermined`, not `same_metric_variant`. The gap between those two is precisely the
recoverability we are measuring.

**The shown table is occasionally the wrong table.** Table selection is now value-anchored (the
extractor's own `table:N`, corroborated by the value actually appearing in that table) and the
snippet is windowed on the value with the header rows kept, so the number you are judging is
visible in the table shown on **88 percent** of sides, up from 57 percent. On the other sides
the value appears in no table of the paper at all; the packet prints a `⚠` line and you should
discount that caption. This affects **20 of the 260 sides**, and it is a genuine ceiling on how
much of the variant is recoverable: where the evidence is not in the corpus, the variant cannot
be read. When the only variant evidence would have come from a different table's caption,
`undetermined` is the correct label.

## Confidence

`confidence_1to5`: 5 = the caption or setup text states the variant outright; 3 = a reasonable
inference from the surrounding text; 1 = a guess. Use `note` for the phrase you relied on.

## What the detector will be given

For the record, so the comparison to your labels is like for like: the detector under test
reads the same things you do — the caption of the value's own table, the table header rows plus
the rows around the reported value, and the setup paragraph. It is **not** given a list of
extracted cue tokens, and it is **not** told which stratum a pair came from. Item 1's ablation
is why: with the table context, arms recovered 30 and 24 of 35 protocol barriers; with cue
tokens alone, 12 of 35. The signal is in the context.

## How the labels will be used

The sample is stratified (enriched on a deterministic context cue, with a no-cue random control
and two contaminated strata as negative controls) and carries post-stratification weights back
to the eligible population of 1,509 pairs. The 130 pairs are split by `dataset_id` into a
50-pair dev fold and an 80-pair eval fold. Only the dev labels will be used to design the
variant-aware prompt; the eval fold is sealed and read once, at scoring. Please label all 130 —
the fold assignment is not shown to you and should not be.

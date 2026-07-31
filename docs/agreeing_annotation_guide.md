# Annotation guide: agreeing pairs (item 1)

Companion to `data/census/agreeing_pairs_packet.md` and
`data/census/agreeing_pairs_sheet.csv` (95 pairs).

## What is different about this task

The released census validated a decision over pairs whose values **differ**. These 95 pairs
**agree**: both papers report the same number for the same canonical (method, dataset, metric)
cell, after the same deterministic percent-versus-fraction reconciliation.

Two facts shape the task. First, **agreement in this literature is copying, not
reproduction**: of all 1,287 agreeing pairs in the corpus, **not one** has both sides reporting
their own result. Two papers independently landing on the identical number essentially never
happens. Second, comparability is a property of protocols, not of values.

So the question this sample really asks is not "does the decision survive when the values
agree". It is:

> **Is the comparability decision driven by the protocol rather than by the value?**

The sharp case is an agreeing-value pair that carries a visible split difference. A
value-driven reading calls those two numbers the same result; a protocol-driven reading still
calls them incomparable. That is why the sample deliberately oversamples exactly those pairs.
For each pair, judge:

> Were these two numbers produced under the same evaluation protocol, as far as the evidence
> shown allows you to tell?

You see each paper's own value, its stated metric and split, the source span, and where
available the table caption, a table snippet, and the setup paragraph. You do **not** see any
external "correct" value, any model prediction, or which stratum the pair came from. Judge only
from what is shown.

## Labels

| Label | Use when |
|---|---|
| `comparable` | As far as the evidence shows, both numbers come from the same evaluation protocol. The agreement is a genuine same-protocol agreement. |
| `split` | The two sides evaluate on different data splits or folds (test vs validation, a different fold, test-dev vs test), so the equal numbers sit across a protocol boundary. |
| `metric_variant` | The same metric name is computed a different way (micro vs macro, filtered vs raw, a different @k, different averaging). A different **spelling** of the same metric (`acc` vs `accuracy`, `mIoU` vs `mean IoU`) is **not** a metric variant. |
| `evaluation_setting` | Some other protocol difference: few-shot vs full, extra training data, input resolution, a different subtask. |
| `citation_copy` | The two entries are the **same underlying number reported twice** — one paper cites the other, or both cite a common source. The agreement carries no independent evidence. |
| `extraction_artifact` | Not a genuine cross-paper pair: an extraction error, or a spurious identity match between two different things. |
| `undetermined` | The evidence shown genuinely does not let you decide. Use this rather than guessing. |

## Priority when more than one applies

    extraction_artifact  >  split / metric_variant / evaluation_setting  >  citation_copy  >  comparable

Rationale: if the pair is not genuine, nothing else matters. If a protocol difference is
visible, that is the finding, whether or not one number was also copied. `citation_copy` beats
`comparable` because an agreement that is really one number counted twice is weaker evidence
than two independent agreeing measurements.

## Notes on the hard cases

**`citation_copy` versus `comparable`.** A cited number that was produced under the same
protocol is both. Label it `citation_copy`: what matters downstream is that the two entries are
not independent. Signals: one side is marked `cited/other`, the span reads like a comparison
table of prior work, or the values match to every digit including an unusual precision.

**`undetermined` is a real answer.** Many pairs will show a value and a span with no protocol
detail at all. If the caption and setup text do not say which split or which metric variant was
used, and you cannot infer it, use `undetermined`. An honest unknown rate is a result; a guessed
label is not.

**Do not use the equality of the values as evidence of comparability.** That is the very
assumption under test. Two papers evaluating on different splits can land on the same number,
and it happens more often than intuition suggests when values are rounded to one decimal.

**Scale.** If the two numbers are on different scales (0.85 vs 85), they were already
reconciled before you saw them, so treat them as equal.

**The shown table is occasionally the wrong table.** Table selection is now value-anchored and
the snippet is windowed on the reported value, so the value you are looking at is visible in
the table shown on **98 percent** of sides (up from 68 percent). On the remaining sides the
value appears in no table of the paper at all; the packet prints a `⚠` line there, and you
should discount that caption and rely on the span and the setup text. This affects **3 of the
190 sides** in this sample. An honest `undetermined` is the right answer when the only protocol
evidence would have come from a different table's caption.

## Confidence

`confidence_1to5`: 5 = the evidence states it outright; 3 = a reasonable reading; 1 = a guess.
Use the `note` field for anything that would change your label if you had more of the paper.

## What happens to these labels

They establish whether the comparability decision follows the protocol or the value. The
cross-protocol stratum (35 pairs, all split differences) is taken **exhaustively** — it is the
whole one-pair-per-cell frame, not a sample of it, so every pair is an independent cell and no
pseudo-replication objection applies. The other two strata carry post-stratification weights
(8.33 and 15.97) back to the population of 1,287 agreeing pairs. Nothing here re-opens the
frozen census gold, and no cell from that gold appears in this sample.

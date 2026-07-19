# Cross-Paper Result Comparability in Machine Learning

A provenance-grounded resource for studying how comparable quantitative results are across
machine-learning papers, built from a frozen Papers with Code (PwC) snapshot dated 2025-07-28.
Companion resource to the JCDL 2026 paper "Are These Numbers Comparable? A Census and
Comparability-Cleaned Leaderboard Resource for Cross-Paper Machine-Learning Results."

## Contents
- `dataset/` -- 3,058 cross-paper result-cell disagreements, each grounded to its two source
  papers and evidence spans, with a beyond-noise decision, a comparability label, and an identity
  grade (`candidates.jsonl`); the noise-model decisions (`noise_decisions.jsonl`); and a blind,
  human-labeled subset of 200 pairs (`gold_sample.jsonl` with `gold_labels.csv`).
- `cleaned_leaderboards/` -- 16,215 ranked entries across 4,438 leaderboards, partitioned into
  comparable clusters with cross-protocol and comparability-unknown entries flagged
  (`cleaned_leaderboards.jsonl`), plus ranking-divergence summaries and a schema.
- `census/` -- the human-validated census with confidence intervals (`census.json`), the
  intra-annotator reliability (`reliability.json`), and the scale-and-cost analysis (`scale_cost.json`).
- `figures/`, `DATASHEET.md`, `MANIFEST.json` (SHA-256 for every file), and `load.py`.

## Key findings
- Of candidate same-cell disagreements, about 45% are real; among the real ones, about 84% are
  protocol artifacts and only about 14% are genuine conflicts.
- Field-wide, comparability is mostly unverifiable: about 60% of head-to-head comparisons cannot be
  confirmed either way, because the protocol detail is not reported in the papers.
- Where comparability can be adjudicated, the naive best-value winner changes on 46% of the
  leaderboards where multiple protocols are demonstrably present.

## Provenance and license
Built from a frozen PwC snapshot dated 2025-07-28, obtained from a public mirror after the live
service was retired on 2025-07-24. Released under CC-BY-SA 4.0. Every file is pinned by SHA-256 in
`MANIFEST.json`.

## Usage
```python
from load import load_dataset, load_cleaned_leaderboards, load_gold
pairs  = load_dataset()               # 3,058 attributed disagreements
boards = load_cleaned_leaderboards()  # 16,215 ranked entries
gold   = load_gold()                  # 200 human-labeled pairs
```
A downstream system can rank within a comparable cluster and abstain on cross-protocol and
comparability-unknown comparisons, keeping its comparisons fair by construction.

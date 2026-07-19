# Cross-Paper Result Comparability in Machine Learning

A provenance-grounded resource for studying whether quantitative results are comparable across
machine-learning papers, built from a frozen Papers with Code (PwC) snapshot dated 2025-07-28.
Companion to the JCDL 2026 Resources paper "Cross-Paper Result Comparability: A Dataset,
Protocol-Partitioned Leaderboards, and a Census."

## What this is, and is not
- Result values and paper identities come from PwC. The result tuples and their locations were
  extracted by our pipeline from the full text of the 1,625 arXiv papers linked in the snapshot.
  This is a curated sample of the ML-leaderboard literature (1,625 of 576,261 snapshot papers), not
  all of machine learning.
- 200 pairs carry HUMAN labels; the remaining pairs carry MODEL-SUGGESTED labels. The `label_source`
  field marks which. Model-suggested labels are not gold and should not be treated as such.

## Contents
- `dataset/comparekg_candidates.jsonl` -- 3,058 cross-paper result-cell disagreements: the canonical
  cell, both sides' values and pointers (arXiv id, URL, section or table location), a beyond-noise
  decision, a model-suggested comparability label, an identity grade, and label-provenance fields.
- `dataset/comparekg_gold.jsonl` -- the 200 human-labeled pairs (`human_label`,
  `model_suggested_label`, `label_source`, `human_validated`, confidence, and second-annotator and
  test-retest flags).
- `cleaned_leaderboards/` -- 16,215 entries across 4,438 leaderboards, partitioned into same-observed-protocol
  clusters, with cross-protocol and comparability-unknown entries flagged rather than silently ranked.
- `census/`, `figures/`, `DATASHEET.md`, `LICENSES.md`, `MANIFEST.json`, `load.py`.

## Evidence text
We do not redistribute excerpt text. Each row points to the source (arXiv id, URL, section or table
location) and the reported value; consult the cited arXiv paper for the text.

## Selected findings (see the paper and census/)
- Of candidate disagreements, about 45% are real; among the real ones, about 84% are protocol
  artifacts and about 14% are genuine conflicts.
- Field scale: about 60% of head-to-head comparisons cannot be verified either way, because the
  protocol detail is not reported.
- Where verifiable, the naive best-value winner changes on 8% of analyzable leaderboards and on 46%
  of the subset where multiple protocols are visible.

## License
The entire release is under CC-BY-SA 4.0; no third-party paper text is redistributed. See LICENSES.md.

## Usage
```python
from load import load_dataset, load_gold, load_cleaned_leaderboards
pairs  = load_dataset()               # 3,058 candidate pairs (model-suggested labels)
gold   = load_gold()                  # 200 human-labeled pairs
boards = load_cleaned_leaderboards()  # 16,215 partitioned entries
```

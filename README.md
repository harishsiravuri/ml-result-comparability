# Cross-Paper Result Comparability in Machine Learning

[![DOI](https://zenodo.org/badge/1305426904.svg)](https://doi.org/10.5281/zenodo.21715275)

A provenance-linked resource for studying whether quantitative results are comparable across
machine-learning papers, built from a frozen Papers with Code (PwC) snapshot dated 2025-07-28.
Companion to the JCDL 2026 Resources paper "Cross-Paper Result Comparability: A Dataset,
Protocol-Partitioned Leaderboards, and a Census."

## What this is, and is not
- Result values and paper identities come from PwC. The result tuples and their locations were
  extracted by our pipeline from the full text of the 1,625 arXiv papers linked in the snapshot.
  This is a curated sample of the ML-leaderboard literature (1,625 of 576,261 snapshot papers), not
  all of machine learning.
- 200 pairs (in `comparekg_gold.jsonl`) carry HUMAN labels. Of the 3,058 candidates, 523 carry a
  MODEL-SUGGESTED cause label (`label_source = model`); the remaining 2,535 have no model-attributed
  cause (`model_suggested_label`, `label_source` = null), since no field-wide judge pass was run.
  Model-suggested labels are not gold and should not be treated as such.

## Contents
- `dataset/comparekg_candidates.jsonl` -- 3,058 candidate cross-paper result-cell disagreement pairs: the canonical
  cell, both sides' values and pointers (arXiv id, URL, section or table location), a beyond-noise
  decision, a model-suggested comparability label where present (523 of 3,058; not gold), an identity grade, and label-provenance fields.
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
pairs  = load_dataset()               # 3,058 candidate pairs (523 model-suggested)
gold   = load_gold()                  # 200 human-labeled pairs
boards = load_cleaned_leaderboards()  # 16,215 partitioned entries
```

## Maintenance and versioning

This is a frozen, versioned release (see `CITATION.cff` and the `v1.0.0` tag). The underlying data
is tied to the 2025-07-28 Papers with Code snapshot and does not change. Corrections are tracked as
errata in the repository issues and released as new tags; the frozen snapshot itself is not modified.

## Citation

Please cite the accompanying JCDL 2026 Resources paper and this release. Machine-readable metadata is in `CITATION.cff`. Archival (version) DOI: 10.5281/zenodo.21715276; concept DOI (all versions): 10.5281/zenodo.21715275.
> Siravuri, H. V. & Alhoori, H. (2026). Cross-Paper Result Comparability: A Dataset, Protocol-Partitioned Leaderboards, and a Census (Version v1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.21715276


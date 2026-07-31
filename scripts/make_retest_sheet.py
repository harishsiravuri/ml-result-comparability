"""Generate the intra-annotator TEST-RETEST sheet from gold_sample.jsonl (in_test_retest
flags). Run this ONLY AFTER the first pass (gold_annotation_sheet.csv) is complete and at
least one week has elapsed, so the first pass treats all 200 pairs identically and the
test-retest Cohen kappa is not optimistically biased. The retest pairs are presented in a
shuffled order (seed) so position cues do not aid recall."""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.paths import CENSUS  # noqa: E402

RETEST_ORDER_SEED = 99  # shuffles presentation order only; membership is fixed in the data


def main() -> None:
    rows = [json.loads(l) for l in open(CENSUS / "gold_sample.jsonl") if l.strip()]
    retest = [r["pair_id"] for r in rows if r.get("in_test_retest")]
    random.Random(RETEST_ORDER_SEED).shuffle(retest)
    out = CENSUS / "gold_annotation_sheet_retest.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "label", "confidence_1to5", "note"])
        for pid in retest:
            w.writerow([pid, "", "", ""])
    print(f"wrote {out} ({len(retest)} retest pairs, presentation order shuffled seed "
          f"{RETEST_ORDER_SEED})")
    print("Re-label these WITHOUT consulting the first-pass labels.")


if __name__ == "__main__":
    main()

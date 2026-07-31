"""Render the blind gold packet (gold_sample.jsonl) into a human-readable markdown
labeling document, so the author can scroll pairs alongside gold_annotation_sheet.csv.
Stays blind: shows only what the packet contains (no model prediction / noise decision /
curated value)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.paths import CENSUS  # noqa: E402


def _side(tag: str, s: dict, ctx: dict) -> str:
    out = [f"**Paper {tag}** ({s['paper_id']}) | value=**{s['value']}**"
           f"{(' ' + s['unit']) if s.get('unit') else ''} | split={s.get('split') or 'unstated'}"
           f" | {'own result' if s.get('is_own_result') else 'cited/other'}",
           f"> span: {(s.get('evidence_quote') or '').strip()[:500]}"]
    if ctx and ctx.get("available"):
        if ctx.get("table_caption"):
            out.append(f"> table caption: {ctx['table_caption'][:300]}")
        if ctx.get("setup_paragraph"):
            out.append(f"> setup: {ctx['setup_paragraph'][:400]}")
    return "\n".join(out)


def main() -> None:
    rows = [json.loads(l) for l in open(CENSUS / "gold_sample.jsonl") if l.strip()]
    lines = ["# Gold labeling packet (blind) — 200 pairs",
             "",
             "Label each pair in `data/census/gold_annotation_sheet.csv` with ONE of: "
             "`split`, `metric_variant`, `evaluation_setting`, "
             "`citation_reporting_discrepancy`, `genuine_conflict`, `extraction_artifact`, "
             "`within_noise`. See `docs/annotation_guide.md`.", ""]
    # First pass treats all 200 pairs identically: NO retest / second-annotator marker is
    # rendered (that membership lives only in gold_sample.jsonl), so the test-retest kappa
    # is not optimistically biased.
    for i, r in enumerate(rows, 1):
        c = r["cell"]
        lines += [
            f"## {i}. `{r['pair_id']}`",
            f"cell: **{c['method']}** on **{c['dataset']}**, metric **{c['metric']}** "
            f"({c['metric_direction']})",
            "",
            _side("A", r["A"], r.get("context_A", {})),
            "",
            _side("B", r["B"], r.get("context_B", {})),
            "",
            "label: ____________  confidence(1-5): ___  note: ____________",
            "", "---", "",
        ]
    out = CENSUS / "gold_packet_readable.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out} ({len(rows)} pairs)")


if __name__ == "__main__":
    main()

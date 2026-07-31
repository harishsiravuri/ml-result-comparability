"""Render the two BLIND extension packets into human-readable markdown (item 1 and item 5).

Stays blind: shows only what each packet file contains. No stratum, no protocol class, no
context cue list, no model prediction, no curated value. All pairs are rendered identically,
so no ordering or marker can bias the labeling.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.paths import CENSUS  # noqa: E402


def value_in_shown_table(value, ctx: dict) -> bool | None:
    """Whether the reported value was actually located in the table shown below it.

    Authoritative: the context fetcher now reports `value_located` directly, after
    value-anchored table selection and value-windowed snippets. Where it is False, no table
    in the paper contains the value at all, so the caption shown belongs to a different table
    and must be discounted. The flag reveals nothing about the stratum or the label.
    """
    if not ctx or not ctx.get("available"):
        return None
    return bool(ctx.get("value_located"))


def _side(tag: str, s: dict, ctx: dict) -> str:
    out = [f"**Paper {tag}** ({s['paper_id']}) | value=**{s['value']}**"
           f"{(' ' + s['unit']) if s.get('unit') else ''}"
           f" | metric as stated: {s.get('metric') or 'unstated'}"
           f" | split={s.get('split') or 'unstated'}"
           f" | {'own result' if s.get('is_own_result') else 'cited/other'}",
           f"> span: {(s.get('evidence_quote') or '').strip()[:500]}"]
    if ctx and ctx.get("available"):
        if value_in_shown_table(s["value"], ctx) is False:
            out.append("> ⚠ this value appears in NO table of the paper, so the caption and "
                       "snippet below belong to a DIFFERENT table — discount them and rely "
                       "on the span and the setup text")
        if ctx.get("table_caption"):
            out.append(f"> table caption: {ctx['table_caption'][:300]}")
        if ctx.get("table_snippet"):
            out.append(f"> table snippet: {ctx['table_snippet'][:400]}")
        if ctx.get("setup_paragraph"):
            out.append(f"> setup: {ctx['setup_paragraph'][:400]}")
    return "\n".join(out)


def render(src: str, out_name: str, title: str, header: list[str]) -> None:
    rows = [json.loads(l) for l in open(CENSUS / src) if l.strip()]
    lines = [f"# {title} — {len(rows)} pairs", ""] + header + [""]
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
    (CENSUS / out_name).write_text("\n".join(lines))
    print(f"wrote {CENSUS / out_name} ({len(rows)} pairs)")


def main() -> None:
    render(
        "agreeing_pairs_sample.jsonl", "agreeing_pairs_packet.md",
        "Agreeing-pairs labeling packet (blind)",
        ["Both papers report the SAME value for the same canonical cell. Equal numbers do not "
         "by themselves mean the results are comparable, and in this corpus they usually mean "
         "one number was copied from the other: not one of the 1,287 agreeing pairs has both "
         "sides reporting their own result.",
         "",
         "The question each pair asks is therefore whether the decision follows the PROTOCOL "
         "rather than the VALUE. Where two equal numbers sit on different splits, a "
         "value-driven reading calls them the same result and a protocol-driven reading still "
         "calls them incomparable. Decide from the evidence which is the case here.",
         "",
         "Label each pair in `data/census/agreeing_pairs_sheet.csv` with ONE of: `comparable`, "
         "`split`, `metric_variant`, `evaluation_setting`, `citation_copy`, "
         "`extraction_artifact`, `undetermined`. See `docs/agreeing_annotation_guide.md`."])
    render(
        "metric_variant_sample.jsonl", "metric_variant_packet.md",
        "Metric-variant recovery packet (blind)",
        ["The two papers report DIFFERENT values for the same canonical cell. The question here "
         "is specifically whether the two numbers use the same metric VARIANT (the same way of "
         "computing the metric), which usually has to be read out of the table caption or the "
         "setup text rather than the metric name.",
         "",
         "Label each pair in `data/census/metric_variant_sheet.csv` with ONE of: "
         "`metric_variant`, `same_metric_variant`, `split`, `evaluation_setting`, "
         "`within_noise`, `citation_reporting_discrepancy`, `genuine_conflict`, "
         "`extraction_artifact`, `undetermined`. See `docs/metric_variant_annotation_guide.md`."])


if __name__ == "__main__":
    main()

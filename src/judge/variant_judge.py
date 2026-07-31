"""Item 5: the metric-variant detector (cached LLM, temperature 0).

DESIGN CONSTRAINT (strategic ruling 2026-07-22, from item 1's ablation). The model reads the
TABLE, not a cue list. Item 1 measured barrier recovery of 32/35 with the full table context
against 12/35 with the deterministic cue block alone, and cue-only agreed with fully bare at
leaf kappa 0.782 — isolated cue strings carry almost no protocol signal. The metric-variant
axis is the one the audit showed lives entirely in the context, so feeding cue tokens here
would test the wrong thing.

Concretely, the `context` arm receives, per side: the caption of the value's own table, the
value-windowed table snippet (header rows plus the rows around the reported value), and the
setup paragraph — at budgets large enough that none of it is clipped away. The context cue that
drove SAMPLING is never shown to the model: enrichment is a sampling device, not an input.

The three arms mirror item 1 exactly, so the same contrast is measurable on the variant axis:

    context     full value-windowed table context (the design under test)
    cues_only   the deterministic cue strings only, no table
    bare        neither
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from common.llm import CachedLLM, parse_json_response
from judge.rules import rule_label

VARIANT_LABELS = ["metric_variant", "same_metric_variant", "split", "evaluation_setting",
                  "within_noise", "citation_reporting_discrepancy", "genuine_conflict",
                  "extraction_artifact", "undetermined"]

# Generous, so the value's rows survive into the prompt. The agreeing judge clipped the
# snippet to 500 characters, which silently undid part of the value-windowed retrieval.
CAPTION_CHARS, SNIPPET_CHARS, SETUP_CHARS = 500, 1400, 800

_DEC_RE = re.compile(r'"label"\s*:\s*"([^"]+)"')
_CONF_RE = re.compile(r'"confidence"\s*:\s*([0-9.]+)')
_EVID_RE = re.compile(r'"evidence"\s*:\s*"([^"]*)"')

_TAXONOMY = """Choose exactly one label:
- "metric_variant": the two sides compute the same metric NAME a different way, and that is the operative difference. Examples: micro vs macro averaging, filtered vs raw MRR, Hits@1 vs Hits@10, class-averaged vs instance-averaged, mAP at a different IoU threshold.
- "same_metric_variant": the evidence shows both sides use the SAME metric variant, so the variant is not what separates the numbers. Use this only when the evidence actually shows it.
- "split": the operative difference is the data split or fold.
- "evaluation_setting": some other protocol choice: few-shot vs full, extra training data, input resolution, a different subtask.
- "within_noise": the values are close enough that run-to-run variation plausibly explains the gap.
- "citation_reporting_discrepancy": meant to be the same result, but one was mis-copied or taken from a different cell.
- "genuine_conflict": same protocol, independently produced, still differ beyond noise.
- "extraction_artifact": not a genuine cross-paper pair.
- "undetermined": the evidence shown does not let you decide.

A metric variant changes how the score is COMPUTED FROM THE SAME PREDICTIONS (which averaging, which cutoff, which filtering). An evaluation setting changes WHAT WAS RUN. If recomputing the metric on the same predictions would remove the difference, it is a variant.
A different SPELLING of the same metric (acc vs accuracy, mIoU vs mean IoU, F-score vs F-measure) is NOT a metric variant.
If the text says nothing about how the metric was computed, answer "undetermined" rather than guessing "same_metric_variant"."""

_SYSTEM = (
    "You are an expert machine-learning methodology reviewer. Two papers report DIFFERENT "
    "numeric results for what looks like the same (method, dataset, metric) result cell. Your "
    "task is specifically to decide whether the two numbers use the same metric VARIANT, and "
    "if not, whether the variant is what separates them. The metric name itself is usually "
    "uninformative: the variant is stated, if anywhere, in the table caption, the table header "
    "or the experimental setup text. Read those. You see each paper's OWN reported value and "
    "text; you do NOT see any external 'ground-truth' value.\n\n" + _TAXONOMY +
    "\n\nRespond with STRICT JSON only, keys in exactly this order.\n"
    '{"label": "<one of the nine>", "confidence": <0.0-1.0>, "evidence": "<<=20 words quoting '
    'the caption/header/setup phrase you relied on, or \'none\'>"}'
)


@dataclass
class VariantOutput:
    label: str
    confidence: float
    evidence: str
    backbone: str
    arm: str
    cached: bool
    cost_usd: float = 0.0
    error: str | None = None


def _fmt_side(tag: str, side: dict, ctx: dict | None, with_context: bool) -> str:
    lines = [
        f"[{tag}] paper={side['paper_id']}  reported_value={side['value']}"
        f"{(' ' + side['unit']) if side.get('unit') else ''}",
        f"     metric as stated: {side.get('metric') or 'unstated'}   "
        f"split as stated: {side.get('split') or 'unstated'}   "
        f"source: {side.get('source_block')}",
        f"     evidence_span: {(side.get('evidence_quote') or '').strip()[:400]}",
    ]
    if with_context and ctx and ctx.get("available"):
        if not ctx.get("value_located"):
            lines.append("     NOTE: this value appears in no table of the paper, so any "
                         "caption below is from a different table and is not evidence.")
        if ctx.get("table_caption"):
            lines.append(f"     table_caption: {ctx['table_caption'][:CAPTION_CHARS]}")
        if ctx.get("table_snippet"):
            lines.append("     table (header rows plus the rows around the reported value): "
                         f"{ctx['table_snippet'][:SNIPPET_CHARS]}")
        if ctx.get("setup_paragraph"):
            lines.append(f"     setup_context: {ctx['setup_paragraph'][:SETUP_CHARS]}")
    return "\n".join(lines)


def build_prompt(pair: dict, ctx_left=None, ctx_right=None, *, arm: str = "context"):
    with_context = arm == "context"
    user = (
        f"CANONICAL CELL\n  method: {pair['method_canonical']}\n"
        f"  dataset: {pair['dataset_canonical']}\n"
        f"  metric: {pair['metric_canonical']} (direction: {pair['metric_direction']})\n\n"
        f"THE TWO REPORTED RESULTS (the papers' own numbers; no external truth value given)\n"
        f"{_fmt_side('A', pair['left'], ctx_left, with_context)}\n"
        f"{_fmt_side('B', pair['right'], ctx_right, with_context)}\n\n"
    )
    if arm == "cues_only":
        r = rule_label(pair)
        mv, sp = r["metric_variant"], r["split"]
        user += ("DETERMINISTIC CUES (no table text)\n"
                 f"- metric surface differs={mv['surface_differs']}; variant tokens "
                 f"A={mv['variant_tokens_left']} B={mv['variant_tokens_right']} "
                 f"(variant_token_differs={mv['variant_token_differs']})\n"
                 f"- split families: A={sp['left_split_family']} B={sp['right_split_family']} "
                 f"(differs={sp['differs']})\n"
                 f"- distinct PwC leaderboards on this dataset+metric: "
                 f"{r.get('n_protocols_on_dataset_metric', 0)}\n\n")
    return _SYSTEM, user + "Return the strict JSON described in the system message."


def _coerce(parsed: dict) -> tuple[str, float, str]:
    lab = str(parsed.get("label", "")).strip().lower()
    if lab not in VARIANT_LABELS:
        lab = "undetermined"
    try:
        conf = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return lab, conf, str(parsed.get("evidence", ""))[:300]


def _recover(text: str):
    m = _DEC_RE.search(text or "")
    if not m:
        return None
    return _coerce({"label": m.group(1),
                    "confidence": (_CONF_RE.search(text).group(1)
                                   if _CONF_RE.search(text) else 0.0),
                    "evidence": (_EVID_RE.search(text).group(1)
                                 if _EVID_RE.search(text) else "")})


async def judge_variant(llm: CachedLLM, pair: dict, *, backbone_cfg: dict,
                        arm: str = "context", ctx_left=None, ctx_right=None) -> VariantOutput:
    system, user = build_prompt(pair, ctx_left, ctx_right, arm=arm)
    try:
        res = await llm.complete(
            model_id=backbone_cfg["model_id"], system=system, user=user,
            temperature=backbone_cfg.get("temperature", 0.0),
            max_tokens=backbone_cfg.get("max_tokens", 1200),
            stage=f"variant_{arm}",
            price_in_per_m=backbone_cfg.get("price_in_per_m", 0.0),
            price_out_per_m=backbone_cfg.get("price_out_per_m", 0.0),
        )
    except Exception as e:  # noqa: BLE001
        return VariantOutput("undetermined", 0.0, "", backbone_cfg["model_id"], arm,
                             cached=False, error=str(e)[:200])
    err = None
    try:
        lab, conf, ev = _coerce(parse_json_response(res.text))
    except (ValueError, KeyError):
        rec = _recover(res.text)
        if rec is None:
            return VariantOutput("undetermined", 0.0, "", backbone_cfg["model_id"], arm,
                                 cached=res.cached, cost_usd=res.cost_usd,
                                 error="parse: unrecoverable")
        lab, conf, ev = rec
        err = "parse: recovered_by_regex"
    return VariantOutput(lab, conf, ev, backbone_cfg["model_id"], arm,
                         cached=res.cached, cost_usd=res.cost_usd, error=err)

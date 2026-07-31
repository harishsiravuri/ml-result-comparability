"""Item 1: the comparability judge for AGREEING pairs (cached LLM, temperature 0).

The frozen census judge classifies the CAUSE OF A DISAGREEMENT, which is undefined when the
two values are equal. This variant asks the question the framework actually poses: two papers
report the SAME number for the same canonical cell, so are those numbers COMPARABLE (same
evaluation protocol), or does the agreement sit across a protocol boundary?

Label space, and the priority the human annotation guide uses when more than one applies:
    extraction_artifact > protocol_artifact > citation_copy > comparable

Everything else is inherited unchanged from the frozen judge: the same context fetcher, the
same deterministic cues, temperature 0, the same sha256 request cache, and the same
no-leakage firewall (the Papers-with-Code curated value is never shown).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from common.llm import CachedLLM, parse_json_response
from judge.rules import rule_label

AGREEING_LABELS = ["comparable", "split", "metric_variant", "evaluation_setting",
                   "citation_copy", "extraction_artifact", "undetermined"]

_DEC_RE = re.compile(r'"decision"\s*:\s*"([^"]+)"')
_SUB_RE = re.compile(r'"sub_type"\s*:\s*(?:"([^"]+)"|null|None)')
_CONF_RE = re.compile(r'"confidence"\s*:\s*([0-9.]+)')

_TAXONOMY = """Choose exactly one decision:
- "comparable": as far as the evidence shows, both numbers were produced under the SAME evaluation protocol, so the agreement is a genuine same-protocol agreement.
- "protocol_artifact": the evaluation protocols demonstrably DIFFER, so the equal numbers do not establish comparability. Then set sub_type to one of:
    - "split": a different data split or fold (e.g. test vs validation, a different fold, test-dev vs test).
    - "metric_variant": the same metric NAME computed a different way (e.g. micro vs macro F1, filtered vs raw MRR, a different @k, different averaging). A different SPELLING of the same metric (acc vs accuracy, mIoU vs mean IoU) is NOT a metric variant.
    - "evaluation_setting": a different setting other than split or metric variant (e.g. few-shot vs full, extra training data, input resolution, a different subtask).
- "citation_copy": the two entries are the SAME underlying number reported twice (one paper cites the other, or both cite a common source), so the agreement carries no independent evidence.
- "extraction_artifact": this is NOT a genuine cross-paper pair (an extraction error, or a spurious identity match between different things).
If more than one applies, use this priority: extraction_artifact, then protocol_artifact, then citation_copy, then comparable."""

_SYSTEM = (
    "You are an expert machine-learning methodology reviewer. Two papers report the SAME "
    "numeric result for what looks like the same (method, dataset, metric) result cell. Equal "
    "numbers do not by themselves establish that the two results are comparable: the papers "
    "may be evaluating under different protocols, or one number may simply be copied from the "
    "other. Decide which. You see each paper's OWN reported value and source text; you do NOT "
    "see any external 'ground-truth' value, and you must not assume one.\n\n" + _TAXONOMY +
    "\n\nRespond with STRICT JSON only, and order the keys exactly as below so the decision "
    "is first. Keep the rationale to at most 25 words.\n"
    '{"decision": "<one of the four>", "sub_type": "<split|metric_variant|evaluation_setting '
    'or null>", "confidence": <0.0-1.0>, "rationale": "<<=25 words citing the evidence>"}'
)


@dataclass
class AgreeingOutput:
    decision: str
    leaf: str                      # decision, with protocol_artifact resolved to its sub_type
    sub_type: str | None
    rationale: str
    confidence: float
    backbone: str
    arm: str
    cached: bool
    cost_usd: float = 0.0
    error: str | None = None


# Budgets large enough that the value-windowed retrieval is not silently undone in the prompt.
# The earlier 300/500/600 clip discarded part of the table the retrieval had just centred on
# the reported value, so the "full context" arm was in fact reading clipped context.
CAPTION_CHARS, SNIPPET_CHARS, SETUP_CHARS = 500, 1400, 800


def _fmt_side(tag: str, side: dict, ctx: dict | None) -> str:
    lines = [
        f"[{tag}] paper={side['paper_id']}  reported_value={side['value']}"
        f"{(' ' + side['unit']) if side.get('unit') else ''}"
        f"  claim={'own result' if side.get('is_own_result') else 'cited/other'}",
        f"     stated_metric={side.get('metric') or 'unstated'}  "
        f"stated_split={side.get('split') or 'unstated'}  source={side.get('source_block')}",
        f"     evidence_span: {(side.get('evidence_quote') or '').strip()[:400]}",
    ]
    if ctx and ctx.get("available"):
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


def build_prompt(pair: dict, ctx_left=None, ctx_right=None,
                 include_cues: bool = True) -> tuple[str, str]:
    """include_cues=False makes the prompt genuinely BARE: the two records and their spans, and
    nothing from the structured layer. That is what the chapter's adversarial baseline means,
    so the arm that claims to be bare must not be handed the deterministic cue block."""
    user = (
        f"CANONICAL CELL\n  method: {pair['method_canonical']}\n"
        f"  dataset: {pair['dataset_canonical']}\n"
        f"  metric: {pair['metric_canonical']} (direction: {pair['metric_direction']})\n\n"
        f"THE TWO REPORTED RESULTS (equal values; the papers' own numbers, no external truth)\n"
        f"{_fmt_side('A', pair['left'], ctx_left)}\n"
        f"{_fmt_side('B', pair['right'], ctx_right)}\n\n"
    )
    if include_cues:
        rules = rule_label(pair)
        sp, mv = rules["split"], rules["metric_variant"]
        cues = [
            f"- split families: A={sp['left_split_family']} B={sp['right_split_family']} "
            f"(both_known={sp['both_known']}, differs={sp['differs']})",
            f"- metric surface differs={mv['surface_differs']}; variant tokens A="
            f"{mv['variant_tokens_left']} B={mv['variant_tokens_right']}",
            f"- distinct PwC leaderboards (protocols) on this dataset+metric: "
            f"{rules.get('n_protocols_on_dataset_metric', 0)}",
            f"- extraction-risk flags: {rules['extraction_risk']}",
            "- the two reported values are EQUAL (after the deterministic percent-vs-fraction "
            "reconciliation); there is no numeric gap to explain",
        ]
        user += "DETERMINISTIC CUES (hints; you decide)\n" + "\n".join(cues) + "\n\n"
    else:
        user += ("The two reported values are EQUAL after a percent-versus-fraction "
                 "reconciliation, so there is no numeric gap to explain.\n\n")
    return _SYSTEM, user + "Return the strict JSON described in the system message."


def _coerce(parsed: dict) -> tuple[str, str, str | None, str, float]:
    dec = str(parsed.get("decision", "")).strip().lower()
    sub = parsed.get("sub_type")
    sub = str(sub).strip().lower() if sub not in (None, "", "null", "none") else None
    rationale = str(parsed.get("rationale", ""))[:600]
    try:
        conf = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    if dec == "protocol_artifact":
        leaf = sub if sub in ("split", "metric_variant", "evaluation_setting") else "undetermined"
    elif dec in ("comparable", "citation_copy", "extraction_artifact"):
        leaf = dec
    else:
        dec, leaf = "undetermined", "undetermined"
    return dec, leaf, sub, rationale, conf


def _recover(text: str):
    m = _DEC_RE.search(text or "")
    if not m:
        return None
    return _coerce({"decision": m.group(1),
                    "sub_type": (_SUB_RE.search(text).group(1) if _SUB_RE.search(text) else None),
                    "confidence": (_CONF_RE.search(text).group(1)
                                   if _CONF_RE.search(text) else 0.0),
                    "rationale": ""})


async def judge_agreeing(llm: CachedLLM, pair: dict, *, backbone_cfg: dict, arm: str = "context",
                         ctx_left=None, ctx_right=None,
                         include_cues: bool = True) -> AgreeingOutput:
    system, user = build_prompt(pair, ctx_left, ctx_right, include_cues)
    try:
        res = await llm.complete(
            model_id=backbone_cfg["model_id"], system=system, user=user,
            temperature=backbone_cfg.get("temperature", 0.0),
            max_tokens=backbone_cfg.get("max_tokens", 1200),
            stage=f"agreeing_{arm}",
            price_in_per_m=backbone_cfg.get("price_in_per_m", 0.0),
            price_out_per_m=backbone_cfg.get("price_out_per_m", 0.0),
        )
    except Exception as e:  # noqa: BLE001 — record failures, never crash the sweep
        return AgreeingOutput("undetermined", "undetermined", None, "", 0.0,
                              backbone_cfg["model_id"], arm, cached=False, error=str(e)[:200])
    err = None
    try:
        dec, leaf, sub, rationale, conf = _coerce(parse_json_response(res.text))
    except (ValueError, KeyError):
        rec = _recover(res.text)
        if rec is None:
            return AgreeingOutput("undetermined", "undetermined", None, "", 0.0,
                                  backbone_cfg["model_id"], arm, cached=res.cached,
                                  cost_usd=res.cost_usd, error="parse: unrecoverable")
        dec, leaf, sub, rationale, conf = rec
        err = "parse: recovered_by_regex"
    return AgreeingOutput(dec, leaf, sub, rationale, conf, backbone_cfg["model_id"], arm,
                          cached=res.cached, cost_usd=res.cost_usd, error=err)

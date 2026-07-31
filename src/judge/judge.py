"""Comparability judge: rules + cached LLM over source spans, cues, and (optional)
fuller table/caption context. Classifies the CAUSE of a beyond-noise numeric
disagreement. NEVER sees the Papers-with-Code curated value (only the papers' own
reported values and text); tests/test_no_leakage enforces the firewall.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from common.llm import CachedLLM, parse_json_response
from census.schema import CAUSE_LABELS, top_level_cause
from judge.rules import rule_label

_TOP_RE = re.compile(r'"top_level_cause"\s*:\s*"([^"]+)"')
_SUB_RE = re.compile(r'"sub_type"\s*:\s*(?:"([^"]+)"|null|None)')
_CONF_RE = re.compile(r'"confidence"\s*:\s*([0-9.]+)')

# Prompt v1 (FROZEN 2026-06-18). The v2 sharpened-boundary pass REGRESSED dev final-cause
# kappa 0.593 -> 0.464 (top-level 0.444 -> 0.245), so it was reverted per the pre-committed
# keep/revert criterion (keep only if >=+0.03 with no regression). The metric_variant vs
# evaluation_setting boundary is handled by the deterministic rule, not extra prompt text.
_TAXONOMY = """Classify the CAUSE of the disagreement into exactly one top_level_cause:
- "protocol_artifact": the two numbers are not comparable because the evaluation protocol differs. Then set sub_type to one of:
    - "split": a different data split or fold (e.g. test vs validation, a different fold, test-dev vs test).
    - "metric_variant": the same metric NAME computed a different way (e.g. micro vs macro F1, filtered vs raw MRR, different @k, different averaging).
    - "evaluation_setting": a different setting other than split or metric variant (e.g. few-shot vs full, extra training data, input resolution, a different subtask).
- "citation_reporting_discrepancy": the two numbers are meant to refer to the SAME result, but one was mis-copied, mis-cited, or taken from a different source cell.
- "genuine_conflict": same protocol, independently produced, and still differ beyond noise.
- "extraction_artifact": this is NOT a real cross-paper disagreement (an extraction error in a reported value, or a spurious identity match between different things)."""

_SYSTEM = (
    "You are an expert machine-learning methodology reviewer. Two papers report different "
    "numeric results for what looks like the same (method, dataset, metric) result cell. "
    "Your job is to classify the CAUSE of the disagreement. You see each paper's OWN "
    "reported value and source text; you do NOT see any external 'ground-truth' value, and "
    "you must not assume one. Judge only from the evidence shown.\n\n" + _TAXONOMY +
    "\n\nRespond with STRICT JSON only, and order the keys exactly as below so the "
    "decision is first. Keep the rationale to at most 25 words.\n"
    '{"top_level_cause": "<one of the four>", "sub_type": "<split|metric_variant|'
    'evaluation_setting or null>", "confidence": <0.0-1.0>, "rationale": "<<=25 words '
    'citing the evidence>"}'
)


@dataclass
class JudgeOutput:
    cause: str                     # LLM-only leaf label in CAUSE_LABELS
    top_level: str
    sub_type: str | None
    rationale: str
    confidence: float
    backbone: str
    context_arm: str               # "lean" | "context"
    rule_label: str | None
    cached: bool
    raw_text: str
    cost_usd: float = 0.0
    error: str | None = None
    final_cause: str = ""          # the JUDGE's output: rule-first (rule_label if it fired) else LLM
    rule_overridden: bool = False

    def __post_init__(self):
        # Rules+LLM combination (PREREGISTRATION Section 5): where a high-confidence
        # deterministic rule fires (a known split difference or a metric-variant token
        # difference), it decides the protocol sub-type; otherwise the LLM decides. The
        # LLM-only `cause` is retained so the LLM component's backbone-robustness is
        # measurable separately.
        if self.rule_label:
            self.final_cause = self.rule_label
            self.rule_overridden = self.rule_label != self.cause
        else:
            self.final_cause = self.cause


def _fmt_side(tag: str, side: dict, ptype_own: bool, ctx: dict | None) -> str:
    lines = [
        f"[{tag}] paper={side['paper_id']}  reported_value={side['value']}"
        f"{(' ' + side['unit']) if side.get('unit') else ''}"
        f"  claim={'own result' if side.get('is_own_result') else 'cited/other'}",
        f"     stated_split={side.get('split') or 'unstated'}  source={side.get('source_block')}",
        f"     evidence_span: {(side.get('evidence_quote') or '').strip()[:400]}",
    ]
    if ctx and ctx.get("available"):
        if ctx.get("table_caption"):
            lines.append(f"     table_caption: {ctx['table_caption'][:300]}")
        if ctx.get("table_snippet"):
            lines.append(f"     table_snippet: {ctx['table_snippet'][:500]}")
        if ctx.get("setup_paragraph"):
            lines.append(f"     setup_context: {ctx['setup_paragraph'][:600]}")
    return "\n".join(lines)


def build_prompt(pair: dict, noise: dict, rules: dict,
                 ctx_left: dict | None = None, ctx_right: dict | None = None) -> tuple[str, str]:
    sp = rules["split"]
    mv = rules["metric_variant"]
    ex = rules["extraction_risk"]
    cue_lines = [
        f"- split families: A={sp['left_split_family']} B={sp['right_split_family']} "
        f"(both_known={sp['both_known']}, differs={sp['differs']})",
        f"- metric surface differs={mv['surface_differs']}; variant tokens A={mv['variant_tokens_left']} "
        f"B={mv['variant_tokens_right']} (variant_token_differs={mv['variant_token_differs']})",
        f"- distinct PwC leaderboards (protocols) on this dataset+metric: "
        f"{rules.get('n_protocols_on_dataset_metric', 0)}",
        f"- extraction-risk flags: {ex}",
        f"- reconciled gap={noise.get('gap')} ({noise.get('decision_units')}); "
        f"beyond-noise threshold={noise.get('threshold')}; range_type={noise.get('range_type')}",
    ]
    user = (
        f"CANONICAL CELL\n  method: {pair['method_canonical']}\n  dataset: {pair['dataset_canonical']}\n"
        f"  metric: {pair['metric_canonical']} (direction: {pair['metric_direction']})\n\n"
        f"THE TWO REPORTED RESULTS (the papers' own numbers; no external truth value given)\n"
        f"{_fmt_side('A', pair['left'], pair['left'].get('is_own_result'), ctx_left)}\n"
        f"{_fmt_side('B', pair['right'], pair['right'].get('is_own_result'), ctx_right)}\n\n"
        f"DETERMINISTIC CUES (hints; you decide the final cause)\n" + "\n".join(cue_lines) +
        "\n\nReturn the strict JSON described in the system message."
    )
    return _SYSTEM, user


def _regex_recover(text: str):
    """Recover (leaf, top, sub, rationale, conf) from a truncated/dirty JSON response.

    The decision keys appear before the (long) rationale, so a truncated object still
    carries them. Returns None if even top_level_cause is absent.
    """
    mt = _TOP_RE.search(text or "")
    if not mt:
        return None
    return _coerce({
        "top_level_cause": mt.group(1),
        "sub_type": (_SUB_RE.search(text).group(1) if _SUB_RE.search(text) else None),
        "confidence": (_CONF_RE.search(text).group(1) if _CONF_RE.search(text) else 0.0),
        "rationale": "",
    })


def _coerce(parsed: dict) -> tuple[str, str, str | None, str, float]:
    top = str(parsed.get("top_level_cause", "")).strip().lower()
    sub = parsed.get("sub_type")
    sub = str(sub).strip().lower() if sub not in (None, "", "null", "none") else None
    rationale = str(parsed.get("rationale", ""))[:600]
    try:
        conf = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    # map to a leaf cause
    if top == "protocol_artifact":
        leaf = sub if sub in ("split", "metric_variant", "evaluation_setting") else "undetermined"
    elif top in ("citation_reporting_discrepancy", "genuine_conflict", "extraction_artifact"):
        leaf = top
    else:
        leaf = "undetermined"
    return leaf, top, sub, rationale, conf


async def judge_pair(llm: CachedLLM, pair: dict, noise: dict, *, backbone_cfg: dict,
                     context_arm: str = "lean", ctx_left=None, ctx_right=None) -> JudgeOutput:
    rules = rule_label(pair)
    system, user = build_prompt(pair, noise, rules, ctx_left, ctx_right)
    try:
        res = await llm.complete(
            model_id=backbone_cfg["model_id"], system=system, user=user,
            temperature=backbone_cfg.get("temperature", 0.0),
            max_tokens=backbone_cfg.get("max_tokens", 1200),
            stage=f"judge_{context_arm}",
            price_in_per_m=backbone_cfg.get("price_in_per_m", 0.0),
            price_out_per_m=backbone_cfg.get("price_out_per_m", 0.0),
        )
    except Exception as e:  # noqa: BLE001 — record failures, do not crash the dev sweep
        return JudgeOutput("undetermined", "undetermined", None, "", 0.0,
                           backbone_cfg["model_id"], context_arm, rules["rule_label"],
                           cached=False, raw_text="", error=str(e)[:200])
    err = None
    try:
        parsed = parse_json_response(res.text)
        leaf, top, sub, rationale, conf = _coerce(parsed)
    except (ValueError, KeyError):
        # Truncated/verbose response: the decision keys appear early, so recover them
        # by regex even from an incomplete JSON object (the rationale is what got cut).
        recovered = _regex_recover(res.text)
        if recovered is None:
            return JudgeOutput("undetermined", "undetermined", None, "", 0.0,
                               backbone_cfg["model_id"], context_arm, rules["rule_label"],
                               cached=res.cached, raw_text=res.text[:1000], cost_usd=res.cost_usd,
                               error="parse: unrecoverable")
        leaf, top, sub, rationale, conf = recovered
        err = "parse: recovered_by_regex"
    if leaf not in CAUSE_LABELS:
        leaf = "undetermined"
    if top not in ("protocol_artifact", "citation_reporting_discrepancy",
                   "genuine_conflict", "extraction_artifact"):
        top = top_level_cause(leaf)
    return JudgeOutput(leaf, top, sub, rationale, conf, backbone_cfg["model_id"],
                       context_arm, rules["rule_label"], cached=res.cached,
                       raw_text=res.text[:1000], cost_usd=res.cost_usd, error=err)

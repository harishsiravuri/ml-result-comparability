"""Baseline (b): FRONTIER-ONLY. A strong model judges inconsistency + cause from the two
papers' TEXT WITHOUT the structured layer (no protocol cues, no deterministic rules, no
leaderboard co-membership, no canonical-identity normalization, no noise decision).

This is the load-bearing rebuttal to "a frontier model could just do it": it is given the
honest raw materials a reader has, and we measure where it wins and loses against the
full rules+structured-layer judge. It runs TWO ways (strategic ruling 2026-06-18):
  controlled  : the SAME model as judge_frontier, with vs without the structured layer;
  adversarial : the STRONGEST available model, bare.
It NEVER sees the Papers-with-Code curated value (firewall test covers src/baselines).
"""

from __future__ import annotations

from common.llm import CachedLLM
from judge.judge import _coerce, _regex_recover  # reuse the same parse/coerce

_SYSTEM = (
    "You are an expert machine-learning methodology reviewer. Two papers report different "
    "numeric results that may or may not refer to the same result. From the text alone, "
    "decide whether they genuinely disagree and, if so, why. Classify into exactly one "
    "top_level_cause:\n"
    "- protocol_artifact (sub_type: split | metric_variant | evaluation_setting)\n"
    "- citation_reporting_discrepancy\n"
    "- genuine_conflict\n"
    "- extraction_artifact (not a real disagreement: an error or a mismatch)\n"
    "Use within_noise as top_level_cause if the gap is within ordinary run-to-run noise.\n"
    "Respond with STRICT JSON, decision keys first, rationale <=25 words:\n"
    '{"top_level_cause": "...", "sub_type": "<...or null>", "confidence": <0-1>, '
    '"rationale": "..."}'
)


def _fmt(tag: str, side: dict, ctx: dict | None) -> str:
    lines = [
        f"[Paper {tag}] reports: method='{side.get('method')}' dataset='{side.get('dataset')}' "
        f"metric='{side.get('metric')}' value={side.get('value')}"
        f"{(' ' + side['unit']) if side.get('unit') else ''} split='{side.get('split') or 'unstated'}'",
        f"   source span: {(side.get('evidence_quote') or '').strip()[:400]}",
    ]
    if ctx and ctx.get("available"):
        if ctx.get("table_caption"):
            lines.append(f"   table caption: {ctx['table_caption'][:300]}")
        if ctx.get("table_snippet"):
            lines.append(f"   table snippet: {ctx['table_snippet'][:500]}")
        if ctx.get("setup_paragraph"):
            lines.append(f"   setup: {ctx['setup_paragraph'][:600]}")
    return "\n".join(lines)


def build_bare_prompt(pair: dict, ctx_left=None, ctx_right=None) -> tuple[str, str]:
    """RAW surface fields only; no canonical IDs, no cues, no rules, no co-membership."""
    user = (
        "Two papers report numeric results that appear related. Judge from the text only.\n\n"
        f"{_fmt('A', pair['left'], ctx_left)}\n{_fmt('B', pair['right'], ctx_right)}\n\n"
        "Return the strict JSON described in the system message."
    )
    return _SYSTEM, user


async def frontier_only_predict(llm: CachedLLM, pair: dict, *, model_cfg: dict,
                                ctx_left=None, ctx_right=None, tag: str = "controlled") -> dict:
    system, user = build_bare_prompt(pair, ctx_left, ctx_right)
    try:
        res = await llm.complete(
            model_id=model_cfg["model_id"], system=system, user=user,
            temperature=model_cfg.get("temperature", 0.0),
            max_tokens=model_cfg.get("max_tokens", 1500),
            stage=f"frontier_only_{tag}",
            price_in_per_m=model_cfg.get("price_in_per_m", 0.0),
            price_out_per_m=model_cfg.get("price_out_per_m", 0.0),
        )
    except Exception as e:  # noqa: BLE001
        return {"pair_id": pair["pair_id"], "method": f"frontier_only_{tag}",
                "model_id": model_cfg["model_id"], "cause": "undetermined",
                "decision_disagreement": None, "error": str(e)[:200]}
    from common.llm import parse_json_response
    try:
        leaf, top, sub, _, conf = _coerce(parse_json_response(res.text))
    except (ValueError, KeyError):
        rec = _regex_recover(res.text)
        leaf, top = (rec[0], rec[1]) if rec else ("undetermined", "undetermined")
        conf = 0.0
    # map "within_noise" top-level to a decision=no
    within = top == "within_noise" or leaf == "within_noise"
    disagreement = not (within or leaf == "extraction_artifact")
    return {"pair_id": pair["pair_id"], "method": f"frontier_only_{tag}",
            "model_id": model_cfg["model_id"], "cause": leaf, "top_level": top,
            "decision_disagreement": disagreement, "confidence": conf,
            "cached": res.cached, "cost_usd": res.cost_usd}

"""Fuller-context retrieval for the comparability judge (read-only paper fulltext).

Given one side of a candidate pair (paper_id, reported value, evidence span, source
block), retrieve a BOUNDED context: the source table caption and a snippet, plus the
nearest experimental-setup paragraph. The cause of a numeric disagreement (filtered vs
raw MRR, test vs val split, few-shot k, input resolution) usually lives in the table
caption/header or the setup paragraph, not the value span; the old Chapter 3 finding
that about 73 percent of fine protocol axes are not reliably extractable is the reason
we expose this richer text and ablate whether it helps.

We read only the PAPERS' OWN text (paper2_contribqa fulltext JSON, read-only) via the
PAPER2_FULLTEXT constant. We never read the Papers-with-Code curated value; the
no-leakage firewall (tests/test_no_leakage) holds because this module touches only the
corpus/fulltext directory, none of the curated-gold or raw-archive directories.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from common.paths import PAPER2_FULLTEXT

_SETUP_CUES = re.compile(
    r"(experiment|evaluation|setup|set-up|implementation|training detail|protocol|"
    r"benchmark|dataset|metric|split|fold|few-shot|zero-shot|resolution|hyperparam)",
    re.I,
)
_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", (s or "")).strip()


@lru_cache(maxsize=2048)
def _load_paper(paper_id: str) -> dict | None:
    p = PAPER2_FULLTEXT / f"{paper_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _value_strings(value: float) -> list[str]:
    """Surface forms of a value to look for in a table (points and fraction scales)."""
    out = set()
    for v in (value, value * 100.0 if abs(value) <= 1.5 else value / 100.0):
        for nd in (1, 2, 3):
            out.add(f"{v:.{nd}f}")
        out.add(str(v))
    return [s for s in out if s and s not in ("0.0", "0.00", "0")]


_TABLE_IDX = re.compile(r"table:(\d+)\s*$", re.I)


def _source_table_index(source_block: str) -> int | None:
    """The table index the extractor recorded for this value, if it recorded one."""
    m = _TABLE_IDX.match(_norm(source_block))
    return int(m.group(1)) if m else None


def _best_table(paper: dict, evidence_quote: str, value: float,
                source_block: str = "") -> tuple[dict | None, bool]:
    """VALUE-ANCHORED table selection. Returns (table, value_located).

    The reported value usually appears in several tables of the same paper (only 103 of 446
    sampled sides had exactly one match; the median side had two or three), so a value match
    alone does not identify the table. The extractor's own `table:N` source block does, when
    that table really contains the value, so it is used as the anchor and the score is only
    the tie-break among value-bearing tables.

    value_located is False when NO table contains the value, which is the case the packet
    must flag rather than quietly show an unrelated caption.
    """
    tables = paper.get("tables") or []
    if not tables:
        return None, False
    vals = _value_strings(value)
    bearing = [t for t in tables if any(v in (t.get("text") or "") for v in vals)]

    idx = _source_table_index(source_block)
    if idx is not None and 0 <= idx < len(tables) and tables[idx] in bearing:
        return tables[idx], True                     # the extractor's anchor, corroborated

    pool = bearing or tables
    eq = _norm(evidence_quote).lower()
    eq_toks = set(re.findall(r"[a-z0-9@]+", eq))
    best, best_score = None, -1.0
    for t in pool:
        text = t.get("text") or ""
        cap = t.get("caption") or ""
        score = float(sum(1 for vs in vals if vs in text))
        if eq_toks:
            score += len(set(re.findall(r"[a-z0-9@]+", cap.lower())) & eq_toks) * 0.2
        if score > best_score:
            best, best_score = t, score
    return best, bool(bearing)


def _setup_paragraph(paper: dict, source_block: str) -> str:
    secs = paper.get("sections") or []
    # prefer a section whose heading carries a setup cue; else the source-block section
    sb = _norm(source_block).lower()
    cand = None
    for s in secs:
        h = (s.get("heading") or "").lower()
        if _SETUP_CUES.search(h):
            cand = s
            break
    if cand is None:
        for s in secs:
            if sb and sb.split(":")[-1].strip()[:18] in (s.get("heading") or "").lower():
                cand = s
                break
    if cand is None:
        return ""
    text = _norm(cand.get("text") or "")
    # return the sentence window around the first setup cue, bounded
    m = _SETUP_CUES.search(text)
    start = max(0, (m.start() if m else 0) - 200)
    return text[start:start + 900]


def _snippet(text: str, value: float, max_chars: int, header_chars: int = 220) -> str:
    """Bounded snippet WINDOWED ON THE VALUE, with the table header kept.

    Taking the first max_chars of a table drops the reported value out of view whenever the
    row sits further down: on the sampled sides the value was in the selected table 98 and 88
    percent of the time but inside a leading 700-character window only 82 and 76 percent. The
    header rows carry the column semantics (which split, which @k), so they are prepended
    rather than replaced.
    """
    text = _norm(text)
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    pos = -1
    for vs in _value_strings(value):
        p = text.find(vs)
        if p != -1 and (pos == -1 or p < pos):
            pos = p
    if pos == -1:
        return text[:max_chars]
    head = text[:header_chars]
    body_budget = max_chars - len(head) - 5
    if pos < len(head):                                   # value already inside the header
        return text[:max_chars]
    start = max(len(head), pos - body_budget // 3)
    return f"{head} ... {text[start:start + body_budget]}"


def fetch_context(paper_id: str, value: float, evidence_quote: str, source_block: str,
                  max_table_chars: int = 700) -> dict:
    """Return {available, value_located, table_caption, table_snippet, setup_paragraph}.

    value_located says whether the reported value was actually found in a table of this
    paper. When it is False the caption shown belongs to some other table, and the caller
    must say so rather than let it be read as the value's own caption.
    """
    paper = _load_paper(paper_id)
    if paper is None:
        return {"available": False, "value_located": False, "table_caption": "",
                "table_snippet": "", "setup_paragraph": ""}
    t, located = _best_table(paper, evidence_quote, value, source_block)
    return {
        "available": True,
        "value_located": bool(located),
        "table_caption": _norm(t.get("caption", "")) if t else "",
        "table_snippet": _snippet(t.get("text", ""), value, max_table_chars) if t else "",
        "setup_paragraph": _setup_paragraph(paper, source_block),
    }

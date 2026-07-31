# PROVENANCE: copied verbatim 2026-06-18 from paper2_contribqa/src/common/llm.py
#   (Paper 2 ContribQA infrastructure (originally adapted from Paper 1 extractor)).
# The source repository is READ-ONLY; this is the working copy for paper3.2
#   (comparekg, Chapter 3 cross-paper result-cell disagreement census).
# Do not edit the source; edit this copy if behavior must change here.
"""Cached, cost-logged OpenRouter client.

Adapted from paper1/src/paper1/openrouter.py (Paper 1 extraction pipeline);
adds a sha256-pinned append-only disk cache and per-call cost logging with a
hard budget guard.

Cache key = sha256 of the canonical JSON of the full request (model, messages,
temperature, max_tokens, top_p, seed). Cache hits cost $0 and never hit the
network, so any crashed job replays for free on --resume.

NO-LEAKAGE INVARIANT: this module never imports gold-generation code or reads
gold/leaderboard files. Prompt content is the caller's responsibility;
tests/test_no_leakage.py enforces the separation statically.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from . import costlog
from .paths import LLM_CACHE, REPO_ROOT

load_dotenv(REPO_ROOT / ".env")


@dataclass
class CompletionResult:
    """One LLM response, plus token-usage metadata for cost tracking."""

    text: str
    tokens_in: int
    tokens_out: int
    model_id: str
    cost_usd: float
    cached: bool
    cache_key: str
    raw: dict[str, Any]


class _RetryableError(Exception):
    pass


class OpenRouterAPIError(Exception):
    pass


def cache_key_for(payload: dict[str, Any]) -> str:
    canon = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


class CachedLLM:
    """Async OpenRouter client with sha256-pinned disk cache and budget guard.

    Prices are passed per call (from config/models.yaml) so cost is computed at
    the moment of spend and appended to experiments/cost_log.jsonl.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_s: float = 180.0,
        max_retries: int = 5,
        cache_dir: Path = LLM_CACHE,
        budget_cap_usd: float = costlog.HARD_CAP_USD,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self._api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        self._cache_dir = cache_dir
        self._budget_cap = budget_cap_usd
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout_s,
        )
        self._max_retries = max_retries
        self._budget_lock = asyncio.Lock()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / key[:2] / f"{key}.json"

    async def complete(
        self,
        *,
        model_id: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        top_p: float = 0.95,
        seed: int | None = 13,
        stage: str = "unspecified",
        price_in_per_m: float = 0.0,
        price_out_per_m: float = 0.0,
    ) -> CompletionResult:
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if seed is not None:
            payload["seed"] = seed
        key = cache_key_for(payload)
        cpath = self._cache_path(key)

        if cpath.exists():
            doc = json.loads(cpath.read_text())
            body = doc["response"]
            usage = body.get("usage", {})
            return CompletionResult(
                text=body["choices"][0]["message"]["content"],
                tokens_in=int(usage.get("prompt_tokens", 0)),
                tokens_out=int(usage.get("completion_tokens", 0)),
                model_id=model_id,
                cost_usd=0.0,
                cached=True,
                cache_key=key,
                raw=body,
            )

        # Budget guard before any network spend.
        async with self._budget_lock:
            costlog.check_budget(cap=self._budget_cap)

        body = await self._post(payload, model_id)
        usage = body.get("usage", {})
        tokens_in = int(usage.get("prompt_tokens", 0))
        tokens_out = int(usage.get("completion_tokens", 0))
        cost = (tokens_in / 1e6) * price_in_per_m + (tokens_out / 1e6) * price_out_per_m

        # Persist cache record FIRST (append-only; never overwrite), then log cost.
        cpath.parent.mkdir(parents=True, exist_ok=True)
        if not cpath.exists():
            tmp = cpath.with_suffix(".tmp")
            tmp.write_text(json.dumps({"request": payload, "response": body}, ensure_ascii=False))
            tmp.rename(cpath)
        async with self._budget_lock:
            costlog.log_call(
                stage=stage,
                model_id=model_id,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost,
                cache_key=key,
            )

        return CompletionResult(
            text=body["choices"][0]["message"]["content"],
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model_id=model_id,
            cost_usd=cost,
            cached=False,
            cache_key=key,
            raw=body,
        )

    async def _post(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=2.0, min=1, max=60),
            retry=retry_if_exception_type((_RetryableError, httpx.TransportError)),
            reraise=True,
        ):
            with attempt:
                resp = await self._client.post("/chat/completions", json=payload)
                if resp.status_code in (408, 429, 500, 502, 503, 504):
                    raise _RetryableError(
                        f"HTTP {resp.status_code} from OpenRouter for '{model_id}': {resp.text[:300]}"
                    )
                if resp.status_code >= 400:
                    raise OpenRouterAPIError(
                        f"HTTP {resp.status_code} for '{model_id}': {resp.text[:500]}"
                    )
                body = resp.json()
                if "choices" not in body or not body["choices"]:
                    # Some providers return 200 with an error object — retry once via marker
                    err = body.get("error", {})
                    raise _RetryableError(f"No choices for '{model_id}': {json.dumps(err)[:300]}")
                if body["choices"][0]["message"].get("content") is None:
                    raise _RetryableError(f"Null content for '{model_id}'")
                return body
        raise RuntimeError("unreachable")  # pragma: no cover

    async def aclose(self) -> None:
        await self._client.aclose()


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse JSON from an LLM response, tolerating fences/preamble/truncation.

    Ported verbatim in behavior from paper1.openrouter.parse_json_response.
    """
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    balanced = _first_balanced_json(text)
    if balanced is not None:
        try:
            return json.loads(balanced)
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    for key in ("contributions", "verdicts", "tuples", "answers"):
        recovered = _recover_truncated_array(text, key)
        if recovered is not None:
            return recovered
    raise ValueError(f"Could not parse JSON from response: {text[:200]!r}")


def _first_balanced_json(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _recover_truncated_array(text: str, key: str) -> dict[str, Any] | None:
    key_idx = text.find(f'"{key}"')
    if key_idx == -1:
        return None
    bracket_idx = text.find("[", key_idx)
    if bracket_idx == -1:
        return None
    objects: list[dict[str, Any]] = []
    i, n = bracket_idx + 1, len(text)
    while i < n:
        while i < n and text[i] in " \t\n\r,":
            i += 1
        if i >= n or text[i] == "]":
            break
        if text[i] != "{":
            break
        depth, in_str, esc, start, end = 0, False, False, i, -1
        while i < n:
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            i += 1
        if end == -1:
            break
        try:
            objects.append(json.loads(text[start:end]))
        except json.JSONDecodeError:
            break
        i = end
    if not objects:
        return None
    return {key: objects}

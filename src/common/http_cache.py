# PROVENANCE: copied verbatim 2026-06-18 from paper2_contribqa/src/common/http_cache.py
#   (Paper 2 ContribQA infrastructure).
# The source repository is READ-ONLY; this is the working copy for paper3.2
#   (comparekg, Chapter 3 cross-paper result-cell disagreement census).
# Do not edit the source; edit this copy if behavior must change here.
"""sha256-pinned HTTP GET cache with polite rate limiting.

Every fetched URL is stored once under data/cache/http/<sha[:2]>/<sha>; a
sidecar .meta.json records url, status, headers subset, and fetch time.
Cache is append-only: existing entries are never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .paths import HTTP_CACHE


def _key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


class CachedHTTP:
    def __init__(
        self,
        *,
        cache_dir: Path = HTTP_CACHE,
        min_interval_s: float = 1.0,
        timeout_s: float = 60.0,
        user_agent: str = "paper2-contribqa-research/0.1 (mailto:harish.siravuri@gmail.com)",
    ) -> None:
        self._cache_dir = cache_dir
        self._min_interval = min_interval_s
        self._last_request = 0.0
        self._client = httpx.Client(
            timeout=timeout_s,
            follow_redirects=True,
            headers={"User-Agent": user_agent},
        )

    def get(self, url: str, *, force_refresh: bool = False) -> tuple[int, bytes]:
        """Return (status_code, body). Serves from cache when present.

        Non-200 responses are cached too (so we don't re-hammer dead URLs);
        callers should check status.
        """
        k = _key(url)
        body_path = self._cache_dir / k[:2] / k
        meta_path = body_path.with_suffix(".meta.json")
        if body_path.exists() and meta_path.exists() and not force_refresh:
            meta = json.loads(meta_path.read_text())
            return int(meta["status"]), body_path.read_bytes()

        wait = self._min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = self._client.get(url)
            status, content = resp.status_code, resp.content
        except httpx.TransportError as e:
            # Do not cache transport failures; surface as status 0
            self._last_request = time.monotonic()
            return 0, str(e).encode()
        self._last_request = time.monotonic()

        body_path.parent.mkdir(parents=True, exist_ok=True)
        if not body_path.exists():
            tmp = body_path.with_suffix(".tmp")
            tmp.write_bytes(content)
            tmp.rename(body_path)
            meta_path.write_text(
                json.dumps(
                    {
                        "url": url,
                        "status": status,
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "fetched_utc": datetime.now(timezone.utc).isoformat(),
                        "content_type": resp.headers.get("content-type", ""),
                    }
                )
            )
        return status, content

    def close(self) -> None:
        self._client.close()

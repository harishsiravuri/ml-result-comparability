"""Load model config (config/models.yaml) for the judge and baselines."""

from __future__ import annotations

from functools import lru_cache

import yaml

from common.paths import CONFIG


@lru_cache(maxsize=1)
def models() -> dict:
    return yaml.safe_load((CONFIG / "models.yaml").read_text())


def model_cfg(name: str) -> dict:
    m = models()
    if name not in m:
        raise KeyError(f"model config '{name}' not in config/models.yaml")
    return m[name]

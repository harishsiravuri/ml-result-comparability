# PROVENANCE: copied verbatim 2026-06-18 from paper2_contribqa/src/common/paths.py
#   (Paper 2 ContribQA infrastructure).
# The source repository is READ-ONLY; this is the working copy for paper3.2
#   (comparekg, Chapter 3 cross-paper result-cell disagreement census).
# Do not edit the source; edit this copy if behavior must change here.
"""Canonical repo paths. Everything resolves relative to the repo root."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA = REPO_ROOT / "data"
RAW = DATA / "raw"
MANIFESTS = DATA / "manifests"
GOLD = DATA / "gold"
CORPUS = DATA / "corpus"
EXTRACTIONS = DATA / "extractions"
INDEX = DATA / "index"
CACHE = DATA / "cache"
LLM_CACHE = CACHE / "llm"
HTTP_CACHE = CACHE / "http"

EXPERIMENTS = REPO_ROOT / "experiments"
RUNS = EXPERIMENTS / "runs"
COST_LOG = EXPERIMENTS / "cost_log.jsonl"

CONFIG = REPO_ROOT / "config"
CHECKPOINTS = REPO_ROOT / "checkpoints"
CENSUS = DATA / "census"

# READ-ONLY references to source repositories (never written to).
# The frozen Papers-with-Code parquet (~1 GB) is NOT duplicated into this repo;
# it is referenced in place and pinned by sha256 in data/manifests + data/SNAPSHOT.md.
# PAPER2_2 is the planned swap target (calibrated trust-scored knowledge graph);
# it does not exist yet and MUST remain a soft, non-blocking dependency.
PAPER2_CONTRIBQA = REPO_ROOT.parent / "paper2_contribqa"
PAPER2_RAW_PWC = PAPER2_CONTRIBQA / "data" / "raw" / "pwc"
# Per-paper fulltext JSON (title/abstract/sections/tables) — the PAPERS' OWN text,
# used read-only for the fuller-context judge arm. This is NOT curated gold; the
# no-leakage firewall (curated PwC values) still holds over anything derived from it.
PAPER2_FULLTEXT = PAPER2_CONTRIBQA / "data" / "corpus" / "fulltext"
PAPER3_COMPLETEQA = REPO_ROOT.parent / "paper3_completeqa"
PAPER2_2 = REPO_ROOT.parent / "paper2.2"  # planned; presence is optional


def ensure_dirs() -> None:
    for p in (RAW, MANIFESTS, GOLD, CORPUS, EXTRACTIONS, INDEX, LLM_CACHE, HTTP_CACHE, RUNS, CENSUS):
        p.mkdir(parents=True, exist_ok=True)

"""Repository paths and read-only pointers to the frozen upstream assets.

Every upstream repository named here is READ ONLY. This module never writes outside
`REPO_ROOT`. Paths are resolved from this file's location so the package works from any
working directory.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PHD_ROOT = REPO_ROOT.parent

# --- read-only upstream assets ------------------------------------------------------
PAPER32_ROOT = PHD_ROOT / "the upstream repository"
PAPER32_SRC = PAPER32_ROOT / "src"
CLEANED_LEADERBOARDS = PAPER32_ROOT / "data" / "cleaned_leaderboards" / "cleaned_leaderboards.jsonl"
CLEANED_LEADERBOARDS_MANIFEST = PAPER32_ROOT / "data" / "cleaned_leaderboards" / "MANIFEST.json"
PWC_PAPERS_MANIFEST = PAPER32_ROOT / "data" / "manifests" / "pwc_papers-with-abstracts.manifest.json"
CH3_NOISE_MODEL = PAPER32_SRC / "noise" / "model.py"

# The sha256 pinned in the upstream MANIFEST at the time this study was built.
CLEANED_LEADERBOARDS_SHA256 = (
    "b2dd904793a6f5dac0815a2936553d67aac2005e2dbb1cdc735fd79073ef2472"
)

# --- writable outputs ---------------------------------------------------------------
DATA = REPO_ROOT / "data"
CHECKPOINTS = REPO_ROOT / "checkpoints"
FIGURES = REPO_ROOT / "figures"
CONFIG = REPO_ROOT / "config"

EXPERIMENTS = REPO_ROOT / "experiments"
LLM_CACHE = EXPERIMENTS / "llm_cache"
COST_LOG = EXPERIMENTS / "cost_log.jsonl"

REPLAY_FAMILIES = DATA / "replay_families.json"
SPLIT_MAP = DATA / "split_map.json"
GATE0_GRID = CHECKPOINTS / "gate0_grid.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ensure_dirs() -> None:
    for d in (DATA, CHECKPOINTS, FIGURES, CONFIG, EXPERIMENTS, LLM_CACHE):
        d.mkdir(parents=True, exist_ok=True)

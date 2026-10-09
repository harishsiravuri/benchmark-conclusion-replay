"""Repository paths and read-only pointers to the frozen upstream assets.

Every upstream repository named here is READ ONLY. This module never writes outside
`REPO_ROOT`. Paths are resolved from this file's location so the package works from any
working directory.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PHD_ROOT = REPO_ROOT.parent

# --- read-only upstream assets ------------------------------------------------------
# The comparability resource this study consumes is a separate, read-only input and is
# not redistributed here (see README). Point UPSTREAM_ROOT at your local copy to rerun
# the phases that need it. Everything that reads only `dataset/` runs without it.
UPSTREAM_ROOT = Path(os.environ.get("UPSTREAM_ROOT", PHD_ROOT / "upstream"))
UPSTREAM_SRC = UPSTREAM_ROOT / "src"
CLEANED_LEADERBOARDS = UPSTREAM_ROOT / "data" / "cleaned_leaderboards" / "cleaned_leaderboards.jsonl"
CLEANED_LEADERBOARDS_MANIFEST = UPSTREAM_ROOT / "data" / "cleaned_leaderboards" / "MANIFEST.json"
PWC_PAPERS_MANIFEST = UPSTREAM_ROOT / "data" / "manifests" / "pwc_papers-with-abstracts.manifest.json"
UPSTREAM_NOISE_MODEL = UPSTREAM_SRC / "noise" / "model.py"

# The sha256 pinned in the upstream MANIFEST at the time this study was built.
CLEANED_LEADERBOARDS_SHA256 = (
    "b2dd904793a6f5dac0815a2936553d67aac2005e2dbb1cdc735fd79073ef2472"
)

# --- writable outputs ---------------------------------------------------------------
# The two frozen inputs the released code needs (replay_families.json, split_map.json)
# ship inside `dataset/`. Fall back to it when the working `data/` tree is absent.
DATA = REPO_ROOT / "data" if (REPO_ROOT / "data").exists() else REPO_ROOT / "dataset"
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

# Reproduction map

Every phase is a pure function of the frozen inputs, the recorded parameters, and seed 13, so
a rerun reproduces the same bytes. Phases 0, 1, 2, and 5 cost nothing. Phase 3 is the only
phase that spends, and its calls are cached by sha256 of the full request, so a rerun of Phase
3 also costs nothing.

## Inputs, all read only

| input | pin |
|---|---|
| `paper3.2/data/cleaned_leaderboards/cleaned_leaderboards.jsonl` | sha256 `b2dd904793a6f5dac0815a2936553d67aac2005e2dbb1cdc735fd79073ef2472`, verified at the start of every script |
| `paper3.2/src/noise/model.py` | imported, not copied; sha256 recorded in every checkpoint |
| `paper3.2/src/census/surface.py`, `paper3.2/src/judge/rules.py` | imported for the split and surface normalisers |
| `paper3.2/data/extractions/tuples.parquet` | Phase 3 only |
| frozen Papers-with-Code papers table | date robustness join only, via `paper3.2/data/manifests/pwc_papers-with-abstracts.manifest.json` |

## Environment

```bash
uv venv --python 3.12 .venv
uv pip install pyarrow numpy scipy pytest pandas python-dotenv httpx tenacity pyyaml matplotlib
```

Phase 3 additionally needs `OPENROUTER_API_KEY` in `.env`. No other phase needs credentials.

## Order of operations

```bash
.venv/bin/python -m pytest tests/ -q                              # 33 tests, includes leakage
.venv/bin/python scripts/phase0_gate0.py                          # Gate 0, family set, split
.venv/bin/python scripts/phase1_replay.py --split all             # replay, all 99
.venv/bin/python scripts/phase1_replay.py --split dev
.venv/bin/python scripts/phase1_replay.py --split test
.venv/bin/python scripts/phase2_metrics.py --split all            # primary analysis
.venv/bin/python scripts/phase2_metrics.py --split dev
.venv/bin/python scripts/phase2_metrics.py --split test           # generalization slice
.venv/bin/python scripts/phase2_sensitivity.py --split all        # seven-variant sweep
.venv/bin/python scripts/phase2_sensitivity.py --split dev
.venv/bin/python scripts/phase5_band_check.py --split all         # noise-band sweep
.venv/bin/python scripts/phase3_propagation.py --stage census     # zero dollars
.venv/bin/python scripts/phase3_propagation.py --stage project    # cost projection, then stop
.venv/bin/python scripts/phase3_propagation.py --stage llm        # the only spending step
.venv/bin/python scripts/phase3_propagation.py --stage analyse
.venv/bin/python scripts/phase5_figures.py                        # four figures
.venv/bin/python scripts/phase5_release.py                        # dataset/ plus MANIFEST
```

## Which artefact carries which claim

| claim in `REPORT.md` | script | artefact |
|---|---|---|
| Gate 0 reproduced, 99 / 45 / 34, six grid rows | `phase0_gate0.py` | `checkpoints/gate0_grid.json` |
| dating coverage and the PwC join | `phase0_gate0.py` | `checkpoints/dating_robustness.json` |
| frozen family list, content hash `c656b4a2...` | `phase0_gate0.py` | `data/replay_families.json` |
| development and held-out split, seed 13 | `phase0_gate0.py` | `data/split_map.json` |
| transition distribution and decomposition (section 4.1) | `phase2_metrics.py --split all` | `checkpoints/metrics_all.json` |
| dual-oracle regret (section 4.2) and the caution negative (4.3) | `phase2_metrics.py --split all` | `checkpoints/metrics_all.json` |
| stability, drift, unknown mass (section 4.4) | `phase2_metrics.py --split all` | `checkpoints/metrics_all.json` |
| seven-variant sensitivity (section 4.5) | `phase2_sensitivity.py --split all` | `checkpoints/sensitivity_all.json` |
| noise-band sweep and the no-reversal gate (section 4.6) | `phase5_band_check.py` | `checkpoints/band_check_all.json` |
| generalization slice (section 5) | `phase2_metrics.py --split test` | `checkpoints/metrics_test.json` |
| deterministic propagation census (section 8.1) | `phase3_propagation.py --stage census` | `checkpoints/propagation_census.json` |
| same-quantity adjudication and qualifier statement (8.2, 8.3) | `phase3_propagation.py --stage analyse` | `checkpoints/propagation_llm_analysis.json` |
| manual instrument check (section 8.4) | read by hand | `data/propagation/adjudication_verdicts.md` |
| cost projection recorded before the first call | `phase3_propagation.py --stage project` | `checkpoints/phase3_cost_projection.json` |
| realised spend | every Phase 3 call | `experiments/cost_log.jsonl` |

## Determinism and what is not reproducible

Deterministic: everything in Phases 0, 1, 2, 5, and the Phase 3a census. Reruns are
byte-identical given the same substrate.

Not exactly reproducible: the Phase 3b model calls. They run at temperature 0 with seed 13,
but a provider may change a model behind an identifier, so a cache miss can return different
text. The cache under `experiments/llm_cache/` is keyed by sha256 of the full request and
makes the recorded run replay exactly; it is what preserves reproducibility, not the seed. If
the cache is discarded, expect small differences in the 797 of 807 and 802 of 807 parse-success
counts and correspondingly small movement in the third decimal place.

The manual adjudication in `data/propagation/adjudication_verdicts.md` is a written record of
one reading of 47 items and is not machine-reproducible. It was performed by the execution
agent rather than an independent human coder, and it is labelled that way.

## Budget

| phase | spend |
|---|---|
| 0, 1, 2, 5 | 0.00 |
| 3 | 3.78 |
| total | 3.78 of a 25 dollar Phase 3 sub-cap, a 40 dollar soft cap, and a 75 dollar hard cap |

The guard in `src/replaykg/common/costlog.py` refuses any non-cached call once cumulative
spend reaches the sub-cap.

# Tracking benchmark rankings with incomplete evaluation conditions

Code and released data for the paper "Tracking Benchmark Rankings with Incomplete
Evaluation Conditions" by Harish Varma Siravuri and Hamed Alhoori, Northern
Illinois University. This repository carries every artefact the paper's claims are
traced to.

## What this is

A leaderboard records what methods scored on a dataset and metric. Readers draw a
comparative conclusion from it, usually which method is best. This repository
replays that conclusion through time. For each benchmark family and each yearly
cutoff, it recomputes the justified conclusion from only the evidence available
at that cutoff, under four update rules, and classifies what happened between
consecutive cutoffs into one of eight typed statuses. No cutoff sees a later
result, and a test fails the run on any leak.

It also contains the citation-propagation measurement: 807 pairs linking an
own-result reading of a canonical cell to a later quoted reading of the same cell
in a different paper, with the model passes that read the citing spans.

## Layout

| path | contents |
|---|---|
| `src/` | the replay, transition classifier, metrics, and propagation code |
| `scripts/` | one script per phase, in the order given in `REPRODUCE.md` |
| `tests/` | the test suite, including the no-future-leakage test |
| `dataset/` | the released resource: replay certificates, transition records, propagation pairs, metrics checkpoints, datasheet and manifest |
| `REPRODUCE.md` | pinned inputs, exact order of operations, and a map from each claim to the artefact that carries it |
| `dataset/DATASHEET.md` | documentation of the released resource |
| `dataset/MANIFEST.json` | sha256 and row count for every released file |

## Reproducing

```bash
uv venv --python 3.12 .venv && uv pip install pyarrow numpy scipy pytest
.venv/bin/python -m pytest tests/ -q
```

The phases that only read `dataset/` run as-is. The phases that rebuild the
replay from the substrate need the upstream comparability resource, which is a
read-only input and is not redistributed here. Point `UPSTREAM_ROOT` at a local
copy to run those:

```bash
UPSTREAM_ROOT=/path/to/upstream .venv/bin/python scripts/phase1_replay.py --split all
```

`REPRODUCE.md` gives the full phase order. Every phase is a pure function of the
frozen inputs and the recorded parameters, with seed 13 throughout, so a rerun
reproduces the same bytes. Two exceptions are stated there: the model calls in
the citation phase replay from a sha256-keyed cache rather than from the seed,
and the manual adjudication is a written record of one reading rather than a
machine-reproducible artefact.

## What is not here

The upstream comparability resource this work consumes is read-only input and is
not redistributed here. The internal annotation sheets used for the blind human
validation are not released, because they carry evidence spans that the source
licence does not permit redistributing. The agreement statistics computed from
them are reported in the paper.

## Licences

The released data in `dataset/` is CC-BY-SA 4.0, matching the licence of the
public archive it is derived from. It is pointers only, with no excerpt text: a
release-time guard aborts the build if any released row carries a span under a
text-bearing key.

The code in `src/`, `scripts/` and `tests/` is MIT licensed. See `LICENSE`.

## Data statement

The substrate is a frozen public snapshot of a community archive of reported
machine-learning results, taken 28 July 2025. Entries span publication years 2014
to 2025. The 99 qualifying benchmark families cover text, knowledge-graph,
document, vision, molecular-property, recommendation and time-series benchmarks;
the per-group membership list is produced by `scripts/phase7_language_subset.py`.
The archive records no personal data and the release carries no excerpt text.

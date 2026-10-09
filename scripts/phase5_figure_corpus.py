"""Phase 5 addendum: fig6_corpus. Zero dollars, deterministic, descriptive only.

Describes the replay substrate. Nothing here is a new measurement or a new claim. Every
value is counted from the released frozen artefact `dataset/replay_certificates.jsonl`,
written by `phase1_replay.py` and pinned in `dataset/MANIFEST.json`. The three panels answer
how much history a qualifying family carries, which the prose states only as thresholds.

  A  results per family at its final cutoff
  B  yearly cutoffs replayed per family
  C  valid comparison sets at each family's final cutoff

Colours are Okabe-Ito, matching the primary figures. Written as PNG at 200 dpi and PDF.
"""

from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
FIGURES = ROOT / "figures"

OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00",
      "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "black": "#000000"}

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False,
})


def load_families():
    by_family = collections.defaultdict(dict)
    with (DATASET / "replay_certificates.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["policy"] != "protocol_partitioned":
                continue
            by_family[row["cell"]["leaderboard_id"]][row["cutoff_year"]] = row
    return by_family


def median_marker(ax, value, label):
    ax.axvline(value, color=OI["black"], lw=0.9, ls="--", alpha=0.6)
    ax.annotate(f"{label} {value:g}", xy=(value, 0.94), xycoords=("data", "axes fraction"),
                xytext=(4, 0), textcoords="offset points", fontsize=7, va="top", alpha=0.8)


def build(by_family):
    finals = {fid: years[max(years)] for fid, years in by_family.items()}
    n_results = sorted(c["evidence_set_size"] for c in finals.values())
    n_cutoffs = sorted(len(years) for years in by_family.values())
    n_sets = collections.Counter(c["comparison_set"]["n_valid_sets"] for c in finals.values())

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.35))

    # A: results per family at the final cutoff
    ax = axes[0]
    ax.hist(n_results, bins=range(0, 330, 15), color=OI["blue"], alpha=0.85, lw=0)
    median_marker(ax, statistics.median(n_results), "median")
    ax.set_xlabel("results in the family at its final cutoff", fontsize=8)
    ax.set_ylabel("families", fontsize=8)
    ax.set_title("A. family size", fontsize=8.5, loc="left")

    # B: replay depth
    ax = axes[1]
    counts = collections.Counter(n_cutoffs)
    ks = sorted(counts)
    ax.bar(ks, [counts[k] for k in ks], color=OI["green"], alpha=0.85, width=0.8, lw=0)
    median_marker(ax, statistics.median(n_cutoffs), "median")
    ax.set_xlabel("yearly cutoffs replayed", fontsize=8)
    ax.set_xticks(ks)
    ax.set_title("B. replay depth", fontsize=8.5, loc="left")

    # C: partition structure at the final cutoff
    ax = axes[2]
    buckets = [("none", n_sets.get(0, 0)),
               ("1", n_sets.get(1, 0)),
               ("2", n_sets.get(2, 0)),
               ("3 or more", sum(v for k, v in n_sets.items() if k >= 3))]
    colours = ["#999999", OI["blue"], OI["purple"], OI["vermillion"]]
    ax.bar([b for b, _ in buckets], [v for _, v in buckets], color=colours, alpha=0.9,
           width=0.7, lw=0)
    for i, (_, v) in enumerate(buckets):
        ax.annotate(str(v), xy=(i, v), xytext=(0, 2), textcoords="offset points",
                    ha="center", fontsize=7.5)
    ax.set_xlabel("valid comparison sets at the final cutoff", fontsize=8)
    ax.set_title("C. partition structure", fontsize=8.5, loc="left")

    for ax in axes:
        ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    return fig


def main():
    by_family = load_families()
    if len(by_family) != 99:
        raise SystemExit(f"expected 99 qualifying families, found {len(by_family)}")
    fig = build(by_family)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"fig6_corpus.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig6_corpus.png and .pdf")


if __name__ == "__main__":
    main()

"""Phase 5 addendum: fig5_worked_example. Zero dollars, deterministic, descriptive only.

Renders one qualifying family, Pubmed accuracy, as a worked example of the replay. Nothing
here is a new measurement. Every value is read from the released frozen artefacts
`dataset/replay_certificates.jsonl` and `dataset/transitions.jsonl`, which were written by
`phase1_replay.py` and are pinned in `dataset/MANIFEST.json`.

The family was selected because it exhibits five of the eight transition statuses in a single
history: unresolved, established, superseded_same_protocol, fragmented_by_protocol_drift, and
weakened. That selection is illustrative and is stated as such in the caption.

Colours are Okabe-Ito, matching the four primary figures. Written as PNG at 200 dpi and PDF.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

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

FAMILY = "pwc-dataset:pubmed|metric:accuracy"

STATUS_SHORT = {
    "unresolved": "unresolved",
    "established": "established",
    "superseded_same_protocol": "superseded\n(same protocol)",
    "fragmented_by_protocol_drift": "fragmented\n(protocol drift)",
    "weakened": "weakened",
    "remains_supported": "remains\nsupported",
    "conditionally_superseded": "conditionally\nsuperseded",
    "contradicted": "contradicted",
}
STATUS_COLOUR = {
    "unresolved": "#888888",
    "established": OI["green"],
    "superseded_same_protocol": OI["blue"],
    "fragmented_by_protocol_drift": OI["purple"],
    "weakened": OI["orange"],
    "remains_supported": OI["green"],
    "conditionally_superseded": OI["sky"],
    "contradicted": OI["vermillion"],
}


def load_family():
    certs, trans = [], []
    with (DATASET / "replay_certificates.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["cell"]["leaderboard_id"] == FAMILY:
                certs.append(row)
    with (DATASET / "transitions.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["family_id"] == FAMILY:
                trans.append(row)
    return certs, trans


def short(name):
    if name is None:
        return "no justified\nconclusion"
    name = name.split(" (")[0]
    return name if len(name) <= 22 else name[:21] + "."


def build(certs, trans):
    naive = {c["cutoff_year"]: c for c in certs if c["policy"] == "naive_latest_max"}
    part = {c["cutoff_year"]: c for c in certs if c["policy"] == "protocol_partitioned"}
    ptrans = {t["from_cutoff"]: t for t in trans if t["policy"] == "protocol_partitioned"}
    years = sorted(part)

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(9.4, 3.5), sharex=True,
        gridspec_kw={"height_ratios": [2.35, 1.0], "hspace": 0.14},
    )

    # ---- top: the two declared-conclusion tracks -------------------------------------
    y_naive, y_part = 1.0, 0.0
    ax.set_ylim(-0.46, 1.42)
    ax.set_yticks([y_part, y_naive])
    ax.set_yticklabels(["protocol-\npartitioned", "naive\nlatest-max"], fontsize=8.5)
    ax.grid(axis="y", alpha=0)

    # shade the stretch where the partitioned policy holds no justified conclusion
    none_years = [y for y in years if part[y]["conclusion"].get("winner_method") is None]
    if none_years:
        ax.axvspan(min(none_years) - 0.45, max(none_years) + 0.45,
                   color="#000000", alpha=0.05, lw=0)

    for track_y, series, colour in ((y_naive, naive, OI["vermillion"]),
                                    (y_part, part, OI["blue"])):
        ax.plot(years, [track_y] * len(years), color=colour, lw=1.4, alpha=0.55, zorder=1)
        prev = "__init__"
        for yr in years:
            cert = series[yr]
            win = cert["conclusion"].get("winner_method")
            val = cert["conclusion"].get("winner_value")
            changed = win != prev
            ax.plot([yr], [track_y], marker="o" if win is not None else "x",
                    ms=7 if changed else 4.5,
                    color=colour if win is not None else "#888888",
                    mfc=colour if changed else "white",
                    mew=1.4, zorder=3)
            if changed:
                label = short(win)
                if val is not None:
                    label += f"\n{val:g}"
                off = 0.15 if track_y == y_naive else -0.15
                va = "bottom" if track_y == y_naive else "top"
                ax.annotate(label, xy=(yr, track_y + off),
                            ha="center", va=va, fontsize=7.2, linespacing=1.05,
                            color=colour if win is not None else "#666666")
            prev = win

    # transition statuses on the partitioned track, drawn between consecutive cutoffs
    for i, yr in enumerate(years[:-1]):
        t = ptrans.get(yr)
        if t is None:
            continue
        mid = yr + 0.5
        col = STATUS_COLOUR.get(t["status"], "#666666")
        ax.annotate("", xy=(yr + 0.92, y_part), xytext=(yr + 0.08, y_part),
                    arrowprops=dict(arrowstyle="-", color=col, lw=2.4, alpha=0.9))
        # stagger consecutive labels so that the long two-line ones do not collide
        height = y_part + (0.13 if i % 2 == 0 else 0.32)
        ax.text(mid, height, STATUS_SHORT.get(t["status"], t["status"]),
                ha="center", va="bottom", fontsize=6.4, color=col, linespacing=0.95)

    ax.set_title(
        "One family replayed: Pubmed, accuracy. The naive track always names a winner; "
        "the protocol-partitioned track names one only once a valid comparison set exists.",
        fontsize=8.5, loc="left")

    # ---- bottom: what the partition is doing underneath ------------------------------
    n_sets = [part[y]["comparison_set"]["n_valid_sets"] for y in years]
    unknown = [part[y]["comparison_set"]["unknown_share"] for y in years]

    bx.step(years, n_sets, where="mid", color=OI["purple"], lw=1.8,
            label="valid comparison sets")
    bx.set_ylabel("valid sets", fontsize=8, color=OI["purple"])
    bx.set_ylim(-0.25, max(n_sets) + 0.85)
    bx.set_yticks(range(0, max(n_sets) + 1))
    bx.tick_params(axis="y", labelcolor=OI["purple"], labelsize=8)

    cx = bx.twinx()
    cx.plot(years, unknown, color=OI["orange"], lw=1.8, marker="o", ms=3.5,
            label="unknown-comparability share")
    cx.set_ylabel("unknown share", fontsize=8, color=OI["orange"])
    cx.set_ylim(-0.05, 1.08)
    cx.tick_params(axis="y", labelcolor=OI["orange"], labelsize=8)
    cx.spines["top"].set_visible(False)
    cx.grid(False)

    # mark the split that produces the fragmentation transition
    split_year = next((t["to_cutoff"] for t in trans
                       if t["policy"] == "protocol_partitioned"
                       and t["status"] == "fragmented_by_protocol_drift"), None)
    if split_year is not None:
        for axis in (ax, bx):
            axis.axvline(split_year, color=OI["purple"], lw=0.9, ls=":", alpha=0.8, zorder=0)
        bx.annotate("the comparable set splits in two",
                    xy=(split_year, max(n_sets)), xytext=(split_year - 3.6, max(n_sets) + 0.45),
                    fontsize=7.2, color=OI["purple"],
                    arrowprops=dict(arrowstyle="->", color=OI["purple"], lw=0.9))

    bx.set_xlabel("cutoff year", fontsize=8.5)
    bx.set_xticks(years)
    bx.set_xticklabels([str(y) for y in years], fontsize=8)
    bx.set_xlim(years[0] - 0.7, years[-1] + 0.7)

    return fig


def main():
    certs, trans = load_family()
    if not certs:
        raise SystemExit(f"family {FAMILY} not found in {DATASET}")
    fig = build(certs, trans)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"fig5_worked_example.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig5_worked_example.png and .pdf")


if __name__ == "__main__":
    main()

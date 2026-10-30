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
    # Single words here. The prose names each status in full, and the parenthetical
    # qualifiers made two adjacent labels wide enough to collide.
    "superseded_same_protocol": "superseded",
    "fragmented_by_protocol_drift": "fragmented",
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
        return "no group with\nmatching conditions"
    name = name.split(" (")[0]
    return name if len(name) <= 22 else name[:21] + "."


def build(certs, trans):
    naive = {c["cutoff_year"]: c for c in certs if c["policy"] == "naive_latest_max"}
    part = {c["cutoff_year"]: c for c in certs if c["policy"] == "protocol_partitioned"}
    ptrans = {t["from_cutoff"]: t for t in trans if t["policy"] == "protocol_partitioned"}
    years = sorted(part)

    # Three panels rather than two. The group count and the ungroupable share were
    # drawn on twin axes of one panel, which asked the reader to track two scales and
    # two colours at once. Each now has its own panel and its own axis.
    fig, (ax, bx, cx) = plt.subplots(
        3, 1, figsize=(9.0, 3.35), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 0.62, 0.62], "hspace": 0.17},
    )

    # ---- top: the two declared-conclusion tracks -------------------------------------
    y_naive, y_part = 1.0, 0.0
    ax.set_ylim(-0.95, 1.95)
    ax.set_yticks([y_part, y_naive])
    ax.set_yticklabels(["checks\nconditions", "ignores\nconditions"], fontsize=10)
    ax.grid(axis="y", alpha=0)

    # shade the stretch where the condition-checking rule has no eligible group
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
                    # One value on this track is recorded on a 0 to 1 scale while the
                    # rest are on 0 to 100. Naming the odd one, with a note for the
                    # others, is the artefact the condition-blind rule walks into.
                    if val <= 1.0:
                        label += "\non a 0 to 1 scale"
                off = 0.17 if track_y == y_naive else -0.17
                va = "bottom" if track_y == y_naive else "top"
                # the first cutoff sits on the axis edge, so its label is left-aligned
                ha = "left" if yr == years[0] else "center"
                ax.annotate(label, xy=(yr - (0.3 if ha == "left" else 0), track_y + off),
                            ha=ha, va=va, fontsize=10.5, linespacing=1.05,
                            color=colour if win is not None else "#666666")
            prev = win

    # Transition statuses on the partitioned track. Consecutive transitions carrying the
    # same status are one episode, which is the unit the paper counts, so each run is
    # drawn and labelled once. Labelling every transition separately crowded the panel
    # and repeated one status five times.
    runs = []
    for yr in years[:-1]:
        t = ptrans.get(yr)
        if t is None:
            continue
        if runs and runs[-1][0] == t["status"] and runs[-1][2] == yr - 1:
            runs[-1][2] = yr
            runs[-1][3] += 1
        else:
            runs.append([t["status"], yr, yr, 1])
    for i, (status, y0, y1, n) in enumerate(runs):
        col = STATUS_COLOUR.get(status, "#666666")
        ax.annotate("", xy=(y1 + 0.92, y_part), xytext=(y0 + 0.08, y_part),
                    arrowprops=dict(arrowstyle="-", color=col, lw=2.6, alpha=0.9))
        label = STATUS_SHORT.get(status, status)
        if n > 1:
            label += f" ({n} steps)" if "\n" not in label else f"\n({n} steps)"
        # Two heights are still needed where a one-step run abuts a longer one. A
        # two-line label starts lower so that its top clears the other track.
        if i % 2 == 0:
            height = y_part + 0.12
        else:
            height = y_part + (0.44 if "\n" not in label else 0.28)
        ax.text((y0 + y1) / 2 + 0.5, height, label,
                ha="center", va="bottom", fontsize=9.2, color=col, linespacing=1.02)

    ax.text(years[-1] + 0.4, y_naive + 0.40, "other values\non a 0 to 100 scale",
            ha="right", va="bottom", fontsize=9.0, color=OI["vermillion"],
            alpha=0.9, linespacing=1.05)

    ax.set_title(
        "One family replayed: Pubmed, accuracy. The condition-blind track always names a winner; "
        "the condition-checking track names one only once a group of two or more exists.",
        fontsize=10, loc="left")

    # ---- bottom: what the partition is doing underneath ------------------------------
    n_sets = [part[y]["comparison_set"]["n_valid_sets"] for y in years]
    unknown = [part[y]["comparison_set"]["unknown_share"] for y in years]

    bx.step(years, n_sets, where="mid", color=OI["purple"], lw=1.8)
    bx.set_ylabel("eligible\ngroups", fontsize=9.5, color=OI["purple"],
                  linespacing=0.95)
    bx.set_ylim(-0.3, max(n_sets) + 0.5)
    bx.set_yticks(range(0, max(n_sets) + 1))
    bx.tick_params(axis="y", labelcolor=OI["purple"], labelsize=8)

    cx.plot(years, unknown, color=OI["orange"], lw=1.8, marker="o", ms=3.5)
    cx.set_ylabel("ungroupable\nshare", fontsize=9.5, color=OI["orange"],
                  linespacing=0.95)
    cx.set_ylim(-0.05, 1.08)
    cx.set_yticks([0.0, 0.5, 1.0])
    cx.tick_params(axis="y", labelcolor=OI["orange"], labelsize=8)

    # One numbered marker for the single event that drives the rest of the history,
    # explained in the text rather than in a floating label on the figure.
    split_year = next((t["to_cutoff"] for t in trans
                       if t["policy"] == "protocol_partitioned"
                       and t["status"] == "fragmented_by_protocol_drift"), None)
    if split_year is not None:
        for axis in (ax, bx, cx):
            axis.axvline(split_year, color="#555555", lw=0.9, ls=":", alpha=0.85,
                         zorder=0)
        bx.annotate("1", xy=(split_year, max(n_sets) + 0.02), ha="center",
                    va="center", fontsize=9, color="#333333", zorder=5,
                    bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="#555555",
                              lw=0.9))

    cx.set_xlabel("cutoff year", fontsize=10)
    cx.set_xticks(years)
    cx.set_xticklabels([str(y) for y in years], fontsize=9.5)
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

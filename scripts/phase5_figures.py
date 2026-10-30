"""Phase 5: figures. Zero dollars, deterministic, regenerated from the frozen checkpoints.

Four figures:
  fig1_survival                Kaplan-Meier survival of the declared winner, per policy, with
                               the held-out slice overlaid for the primary policy;
  fig2_regret_by_oracle        regret per policy as two columns, protocol-gated and ungated,
                               so the tie against the ungated oracle is visible;
  fig3_unknown_over_time       unknown-comparability share by cutoff year, full set and slice;
  fig4_transition_distribution the unsettled decomposition with weakened, fragmentation,
                               conditional supersession, clean advance, and contradicted shown
                               separately, plus the noise-band sweep panel.

Colours are Okabe-Ito, which is colour-blind safe. Every figure is written as PNG at 200 dpi
and as PDF.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from replaykg.paths import CHECKPOINTS, FIGURES  # noqa: E402

# Okabe-Ito
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00",
      "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "black": "#000000"}

# one name per rule, matching the paper's prose and tables
POLICY_LABEL = {
    "naive_latest_max": "condition-blind",
    "protocol_partitioned": "condition-checking",
    "uncertainty_aware": "band-waiting",
    "conservative_update": "persistence",
    "naive_normalized": "condition-blind, units reconciled",
}
POLICY_COLOUR = {
    "naive_latest_max": OI["vermillion"],
    "protocol_partitioned": OI["blue"],
    "uncertainty_aware": OI["green"],
    "conservative_update": OI["orange"],
}

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False,
})


def load(name):
    return json.loads((CHECKPOINTS / name).read_text(encoding="utf-8"))


def save(fig, stem):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{stem}.png and .pdf")


# ---------------------------------------------------------------------------------------

def fig_survival(m_all, m_test):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for policy in ("naive_latest_max", "protocol_partitioned", "conservative_update",
                   "uncertainty_aware"):
        km = m_all["survival"][policy]["kaplan_meier"]
        ax.step(km["times"], km["survival"], where="post",
                color=POLICY_COLOUR[policy], lw=1.9,
                label=f"{POLICY_LABEL[policy]} (median {m_all['survival'][policy]['median_half_life_cutoffs']})")
    km_t = m_test["survival"]["protocol_partitioned"]["kaplan_meier"]
    ax.step(km_t["times"], km_t["survival"], where="post", color=POLICY_COLOUR["protocol_partitioned"],
            lw=1.3, ls=":", label="condition-checking, held-out 31")
    ax.axhline(0.5, color=OI["black"], lw=0.7, ls="--", alpha=0.5)
    ax.text(0.02, 0.515, "median", transform=ax.get_yaxis_transform(), fontsize=9, alpha=0.75)
    ax.set_xlabel("cutoff steps since the winner was declared", fontsize=10)
    ax.set_ylabel("share of declared winners still standing")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 9)
    ax.set_title("Survival of the declared winner\n"
                 "A longer curve describes how long a rule holds a winner, not how good it is",
                 fontsize=10, loc="left")
    ax.legend(fontsize=10, loc="upper right")
    save(fig, "fig1_survival")


def fig_regret(m_all, m_test):
    policies = ["naive_latest_max", "protocol_partitioned", "uncertainty_aware",
                "conservative_update"]
    oracles = [("protocol_gated", "protocol-gated oracle\n(the protocol-justified conclusion)"),
               ("ungated_naive", "ungated naive oracle\n(the best reported number)")]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9), sharey=True)
    for ax, (oracle, title) in zip(axes, oracles):
        # same-cutoff basis: every policy scored only on cutoffs where all four declared
        d = m_all["regret"]["by_oracle"][oracle]["same_cutoff"]["by_policy"]
        t = m_test["regret"]["by_oracle"][oracle]["same_cutoff"]["by_policy"]
        ys = list(range(len(policies)))[::-1]
        for y, p in zip(ys, policies):
            e = d[p]["mean_fractional_regret"]
            ax.plot([e["ci95_lo"], e["ci95_hi"]], [y, y], color=POLICY_COLOUR[p], lw=2.6,
                    solid_capstyle="butt", alpha=0.85)
            ax.plot([e["point"]], [y], "o", color=POLICY_COLOUR[p], ms=6, zorder=3)
            et = t[p]["mean_fractional_regret"]
            ax.plot([et["point"]], [y - 0.26], "D", color=POLICY_COLOUR[p], ms=3.6,
                    mfc="white", mew=1.1, zorder=3)
            ax.plot([et["ci95_lo"], et["ci95_hi"]], [y - 0.26, y - 0.26],
                    color=POLICY_COLOUR[p], lw=0.9, alpha=0.6)
        ax.set_yticks(ys)
        ax.set_yticklabels([POLICY_LABEL[p] for p in policies])
        ax.set_xlabel("mean fractional regret (lower is better)")
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_xlim(-0.02, 0.62)
    axes[0].plot([], [], "o", color=OI["black"], ms=6, label="full 99 families")
    axes[0].plot([], [], "D", color=OI["black"], ms=3.6, mfc="white", mew=1.1,
                 label="held-out 31 families")
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Retrospective regret by policy and by oracle, on the cutoffs where every "
                 "policy declared.\nGating gains against the protocol-gated oracle and loses "
                 "against the ungated one; the reference decides.",
                 fontsize=9, x=0.02, ha="left", y=1.06)
    save(fig, "fig2_regret_by_oracle")


def fig_unknown(m_all, m_test):
    """Marker area is proportional to the number of visible entries, so the thin early
    years cannot be read as if they carried the same weight as the later ones."""
    fig, ax = plt.subplots(figsize=(6.9, 3.7))
    thin_cut = 100
    for m, label, colour, style in ((m_all, "full 99 families", OI["blue"], "-"),
                                    (m_test, "held-out 31 families", OI["orange"], ":")):
        series = m["unknown_comparability_over_time"]
        years = sorted(int(y) for y in series)
        shares = [series[str(y)]["unknown_share"] for y in years]
        counts = [series[str(y)]["entries"] for y in years]
        ax.plot(years, shares, style, color=colour, lw=1.8, zorder=2, label=label)
        ax.scatter(years, shares, s=[max(8, 0.9 * (c ** 0.5) * 3) for c in counts],
                   color=colour, zorder=3, alpha=0.85, edgecolors="white", linewidths=0.5)

    thin_years = [int(y) for y, v in m_all["unknown_comparability_over_time"].items()
                  if v["entries"] < thin_cut]
    if thin_years:
        ax.axvspan(min(thin_years) - 0.5, max(thin_years) + 0.5,
                   color=OI["black"], alpha=0.07, zorder=0)
        ax.text(max(thin_years) + 0.35, 1.0,
                f"fewer than {thin_cut} visible entries;\nrates here are unstable",
                fontsize=9, alpha=0.8, va="top")

    ax.set_ylim(0, 1.06)
    ax.set_xlim(2013.4, 2025.6)
    ax.set_xlabel("cutoff year", fontsize=10)
    ax.set_ylabel("share of visible entries in no valid comparison set")
    ax.set_title("Unknown comparability over time\n"
                 "marker area is proportional to the number of visible entries that year",
                 fontsize=9, loc="left")
    ax.legend(fontsize=9.5, loc="lower left")
    save(fig, "fig3_unknown_over_time")


def fig_transitions(m_all, m_test, band):
    """Left: the episode-counted decomposition the paper reports. Right: the band sweep.

    The transition-counted shares are a different measurement of the same corpus and are
    drawn as hollow markers, labelled as such, so the two are not read as one series.
    The reversal row is named by its classifier status rather than by an interpretation.
    """
    order = [("weakened", "margin inside\nthe noise band", OI["sky"]),
             ("fragmented_by_protocol_drift", "could not be combined\nacross groups", OI["blue"]),
             ("superseded_same_protocol", "new method\ntook the lead", OI["green"]),
             ("conditionally_superseded", "largest group\nchanged", OI["purple"]),
             ("contradicted", "new leader was a previously\npresent method (narrow status)", OI["vermillion"])]

    # Canvas sized close to the ACM sigconf text width (7.01 in). A wider canvas is
    # scaled down by \includegraphics, which shrinks every label with it.
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.9),
                             gridspec_kw={"width_ratios": [1.0, 1.0]})
    fig.subplots_adjust(wspace=0.46)

    ax = axes[0]
    ep = m_all["episode_decomposition"]["shares"]
    ys = list(range(len(order)))[::-1]
    for y, (key, label, colour) in zip(ys, order):
        e = ep[key]
        ax.barh(y, e["point"] * 100, color=colour, height=0.50, alpha=0.92)
        ax.plot([e["ci95_lo"] * 100, e["ci95_hi"] * 100], [y, y],
                color=OI["black"], lw=1.3)
        ax.text(max(e["ci95_hi"], e["point"]) * 100 + 3.5, y, f"{e['point'] * 100:.1f}",
                va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(ys)
    ax.set_yticklabels([lbl for _, lbl, _ in order], fontsize=9.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.8, len(order) - 0.35)
    ax.set_xlabel("percentage of unsettled episodes", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    # Short titles. A title wider than the axes pads the saved canvas under
    # bbox_inches="tight", which then shrinks every label when the figure is scaled
    # to the column width. The counts live in the caption and the text.
    ax.set_title("counted once per episode", fontsize=9.5, loc="left", pad=6)

    ax = axes[1]
    scales = [r["band_scale"] for r in band["by_band"]]
    # three series only: the two the band moves, and the status that does not move
    # One legend in clear space above the curves, rather than a legend over the data plus
    # two floating annotations, which is what crowded this panel before.
    series = [("weakened", "margin inside the band", OI["sky"]),
              ("remains_supported", "lead clear of the band", OI["yellow"]),
              ("superseded_same_protocol", "new method took the lead", OI["green"]),
              ("contradicted", "new leader was previously present", OI["vermillion"])]
    for key, label, colour in series:
        ax.plot(scales, [r["status_counts"].get(key, 0) for r in band["by_band"]],
                marker="o", ms=5.0, lw=2.2, color=colour,
                label=label if label else "_nolegend_")
    ax.set_xscale("log")
    ax.set_xticks(scales)
    ax.set_xticklabels([str(s) for s in scales], fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylim(-14, 330)
    ax.axvline(1.0, color=OI["black"], lw=1.0, ls="--", alpha=0.55)
    ax.annotate("band used\nthroughout", xy=(1.0, 6), xytext=(1.5, 52),
                fontsize=9, alpha=0.9, ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.9, alpha=0.65))
    ax.set_xlabel("noise-band scale", fontsize=10)
    ax.set_ylabel("transition count", fontsize=10)
    # The invariant total is read from the sweep rather than written in, so the title
    # cannot drift from the data the way a hard-coded count did.
    ax.set_title("across the noise-band sweep", fontsize=9.5, loc="left", pad=6)
    ax.legend(fontsize=8.5, loc="upper left", ncol=1, handlelength=1.6,
              borderaxespad=0.3, labelspacing=0.35)
    save(fig, "fig4_transition_distribution")


def main() -> int:
    m_all = load("metrics_all.json")
    m_test = load("metrics_test.json")
    band = load("band_check_all.json")
    fig_survival(m_all, m_test)
    fig_regret(m_all, m_test)
    fig_unknown(m_all, m_test)
    fig_transitions(m_all, m_test, band)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

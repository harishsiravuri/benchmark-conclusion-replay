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

POLICY_LABEL = {
    "naive_latest_max": "naive latest-max",
    "protocol_partitioned": "protocol-partitioned",
    "uncertainty_aware": "uncertainty-aware",
    "conservative_update": "conservative update",
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
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for policy in ("naive_latest_max", "protocol_partitioned", "conservative_update",
                   "uncertainty_aware"):
        km = m_all["survival"][policy]["kaplan_meier"]
        ax.step(km["times"], km["survival"], where="post",
                color=POLICY_COLOUR[policy], lw=1.9,
                label=f"{POLICY_LABEL[policy]} (median {m_all['survival'][policy]['median_half_life_cutoffs']})")
    km_t = m_test["survival"]["protocol_partitioned"]["kaplan_meier"]
    ax.step(km_t["times"], km_t["survival"], where="post", color=POLICY_COLOUR["protocol_partitioned"],
            lw=1.3, ls=":", label="protocol-partitioned, held-out 31")
    ax.axhline(0.5, color=OI["black"], lw=0.7, ls="--", alpha=0.5)
    ax.text(0.02, 0.515, "median", transform=ax.get_yaxis_transform(), fontsize=7, alpha=0.7)
    ax.set_xlabel("cutoffs since the winner was declared")
    ax.set_ylabel("share of declared winners still standing")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 9)
    ax.set_title("Survival of the declared winner\n"
                 "A longer curve describes how long a rule holds a winner, not how good it is",
                 fontsize=9, loc="left")
    ax.legend(fontsize=7.5, loc="upper right")
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
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
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
                fontsize=7, alpha=0.75, va="top")

    ax.set_ylim(0, 1.06)
    ax.set_xlim(2013.4, 2025.6)
    ax.set_xlabel("cutoff year")
    ax.set_ylabel("share of visible entries in no valid comparison set")
    ax.set_title("Unknown comparability over time\n"
                 "marker area is proportional to the number of visible entries that year",
                 fontsize=9, loc="left")
    ax.legend(fontsize=8, loc="lower left")
    save(fig, "fig3_unknown_over_time")


def fig_transitions(m_all, m_test, band):
    order = [("weakened", "weakened\n(margin inside noise band)", OI["sky"]),
             ("fragmented_by_protocol_drift", "fragmented by\nprotocol drift", OI["blue"]),
             ("superseded_same_protocol", "clean same-protocol\nadvance", OI["green"]),
             ("conditionally_superseded", "conditionally\nsuperseded", OI["purple"]),
             ("contradicted", "contradicted\n(empirical reversal)", OI["vermillion"])]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3),
                             gridspec_kw={"width_ratios": [1.0, 1.05]})
    fig.subplots_adjust(wspace=0.42)

    ax = axes[0]
    dec_all = m_all["transitions"]["protocol_partitioned"]["structural_change"][
        "decomposition_among_unsettled"]
    dec_test = m_test["transitions"]["protocol_partitioned"]["structural_change"][
        "decomposition_among_unsettled"]
    ys = list(range(len(order)))[::-1]
    for y, (key, label, colour) in zip(ys, order):
        e = dec_all[key]
        ax.barh(y, e["point"], color=colour, height=0.46, alpha=0.9)
        ax.plot([e["ci95_lo"], e["ci95_hi"]], [y, y], color=OI["black"], lw=1.1)
        et = dec_test[key]
        ax.plot([et["point"]], [y - 0.29], "D", color=colour, ms=4, mfc="white", mew=1.1)
        ax.text(max(e["ci95_hi"], e["point"]) + 0.03, y, f"{e['point']:.3f}",
                va="center", fontsize=8.5, fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels([lbl for _, lbl, _ in order], fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.7, len(order) - 0.4)
    ax.set_xlabel("share of unsettled transitions")
    ax.plot([], [], "D", color=OI["black"], ms=4, mfc="white", mew=1.1,
            label="held-out 31 families")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("Decomposition of unsettled transitions\n"
                 "full 99 families, bars are 95 percent bootstrap intervals",
                 fontsize=9, loc="left", pad=10)

    ax = axes[1]
    scales = [r["band_scale"] for r in band["by_band"]]
    series = list(order) + [("remains_supported", "remains supported", OI["yellow"])]
    for key, label, colour in series:
        ax.plot(scales, [r["status_counts"].get(key, 0) for r in band["by_band"]],
                marker="o", ms=3.6, lw=1.7, color=colour,
                label=label.replace("\n", " "))
    ax.set_xscale("log")
    ax.set_xticks(scales)
    ax.set_xticklabels([str(s) for s in scales], fontsize=7.5)
    ax.set_ylim(-16, 305)
    ax.axvline(1.0, color=OI["black"], lw=0.8, ls="--", alpha=0.5)
    ax.annotate("preregistered\nband", xy=(1.0, 8), xytext=(1.45, 42),
                fontsize=7, alpha=0.85, ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.7, alpha=0.6))
    ax.annotate("contradicted stays at 0\nacross the whole sweep",
                xy=(0.25, -2), xytext=(0.055, 62), fontsize=7.5, color=OI["vermillion"],
                arrowprops=dict(arrowstyle="->", lw=0.8, color=OI["vermillion"]))
    ax.set_xlabel("noise-band scale (1.0 is the preregistered primary band)")
    ax.set_ylabel("transition count")
    ax.set_title("Noise-band sweep\n"
                 "mass moves only between weakened, remains supported and clean "
                 "advance,\nwhich sum to 245 at every band; unresolved (257) and "
                 "established (13) are omitted",
                 fontsize=8.5, loc="left", pad=10)
    ax.legend(fontsize=7, loc="upper center", ncol=2, columnspacing=1.0)
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

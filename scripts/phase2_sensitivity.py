"""Sensitivity of the Gate 1 quantities to the three free parameters. Zero dollars.

The primary analysis fixes three parameters. This script varies each one at a time and
reports the Gate 1 quantities, so the preregistration can record whether the finding depends
on a threshold choice.

  unresolved_threshold  the unknown-comparability share above which a cutoff is unresolved
                        (primary 0.5; a value above 1 disables the rule entirely, so the only
                        route to unresolved is the absence of any valid comparison set);
  conservative_k        consecutive cutoffs a challenger must lead (primary 2);
  band_scale            multiplier on the the upstream resource noise band (primary 1.0).

Writes `checkpoints/sensitivity_<split>.json`.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replaykg.metrics import (  # noqa: E402
    bootstrap_mean, bootstrap_paired_difference, bootstrap_ratio, family_regret,
    ranking_reversals, structural_change_decomposition, supersession_rates,
)
from replaykg.paths import CHECKPOINTS, SPLIT_MAP  # noqa: E402
from replaykg.replay import POLICIES, replay_family  # noqa: E402
from replaykg.substrate import load_families, verify_substrate  # noqa: E402
from replaykg.transitions import family_transitions  # noqa: E402

PRIMARY_POLICY = "protocol_partitioned"

VARIANTS = [
    {"name": "primary", "unresolved_threshold": 0.5, "k": 2, "band": 1.0},
    {"name": "unresolved_threshold_0.3", "unresolved_threshold": 0.3, "k": 2, "band": 1.0},
    {"name": "unresolved_threshold_0.7", "unresolved_threshold": 0.7, "k": 2, "band": 1.0},
    {"name": "unresolved_rule_disabled", "unresolved_threshold": 1.01, "k": 2, "band": 1.0},
    {"name": "conservative_k_3", "unresolved_threshold": 0.5, "k": 3, "band": 1.0},
    {"name": "noise_band_half", "unresolved_threshold": 0.5, "k": 2, "band": 0.5},
    {"name": "noise_band_double", "unresolved_threshold": 0.5, "k": 2, "band": 2.0},
]


def run_variant(families, multi, unresolved_threshold, k, band) -> dict:
    replayed = {f.leaderboard_id: replay_family(f, k=k, band=band) for f in families}

    counts = []
    mp_counts = []
    pooled = collections.Counter()
    for fid, r in replayed.items():
        c = collections.Counter(
            t.status for t in family_transitions(r, PRIMARY_POLICY, unresolved_threshold, band))
        counts.append(dict(c))
        pooled.update(c)
        if fid in multi:
            mp_counts.append(dict(c))

    regret = {p: {} for p in POLICIES}
    ungated = {p: {} for p in POLICIES}
    for fid, r in replayed.items():
        for p in POLICIES:
            fr = family_regret(r, p)
            if fr is not None:
                regret[p][fid] = fr
            fu = family_regret(r, p, "ungated_naive")
            if fu is not None:
                ungated[p][fid] = fu
    scored = sorted(set.intersection(*[set(regret[p]) for p in POLICIES]))
    scored_ungated = sorted(set.intersection(*[set(ungated[p]) for p in POLICIES]))

    rev_multi = []
    for fid, r in replayed.items():
        if fid in multi:
            rr = ranking_reversals(r, PRIMARY_POLICY)
            rev_multi.append((rr["n_with_reversal"], rr["n_transitions"]))

    sc = structural_change_decomposition(counts)
    return {
        "status_counts": dict(pooled),
        "not_clean_advance_among_unsettled": sc["not_clean_advance_among_unsettled"],
        "unsettled_among_resolved": sc["unsettled_among_resolved"],
        "protocol_attributable_among_unsettled": sc["protocol_attributable_among_unsettled"],
        "regret_difference_vs_naive_ungated_oracle": {
            p: bootstrap_paired_difference(
                [(ungated[p][f]["mean_fractional_regret"],
                  ungated["naive_latest_max"][f]["mean_fractional_regret"])
                 for f in scored_ungated]).as_dict()
            for p in POLICIES if p != "naive_latest_max"},
        "protocol_conditional_supersession_among_changes":
            supersession_rates(counts)["among_changes"],
        "protocol_conditional_supersession_multi_protocol":
            supersession_rates(mp_counts)["among_changes"],
        "ranking_reversal_multi_protocol": bootstrap_ratio(rev_multi).as_dict(),
        "mean_fractional_regret": {
            p: bootstrap_mean([regret[p][f]["mean_fractional_regret"] for f in scored]).as_dict()
            for p in POLICIES},
        "regret_difference_vs_naive": {
            p: bootstrap_paired_difference(
                [(regret[p][f]["mean_fractional_regret"],
                  regret["naive_latest_max"][f]["mean_fractional_regret"]) for f in scored]).as_dict()
            for p in POLICIES if p != "naive_latest_max"},
        "regret_difference_vs_protocol_partitioned": {
            p: bootstrap_paired_difference(
                [(regret[p][f]["mean_fractional_regret"],
                  regret[PRIMARY_POLICY][f]["mean_fractional_regret"]) for f in scored]).as_dict()
            for p in ("uncertainty_aware", "conservative_update")},
        "n_families_scored_for_regret": len(scored),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    args = ap.parse_args()

    digest = verify_substrate()
    assignment = json.loads(SPLIT_MAP.read_text(encoding="utf-8"))["assignment"]
    wanted = {k for k, v in assignment.items() if args.split == "all" or v == args.split}
    families = sorted((f for f in load_families() if f.leaderboard_id in wanted),
                      key=lambda f: f.leaderboard_id)
    multi = {f.leaderboard_id for f in families if f.is_multi_protocol}

    out = {"split": args.split, "substrate_sha256": digest, "spend_usd": 0.0, "variants": {}}
    for v in VARIANTS:
        out["variants"][v["name"]] = {
            "parameters": {k: v[k] for k in ("unresolved_threshold", "k", "band")},
            **run_variant(families, multi, v["unresolved_threshold"], v["k"], v["band"]),
        }
        print(f"done {v['name']}")

    path = CHECKPOINTS / f"sensitivity_{args.split}.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

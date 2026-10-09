"""Gate 1 and Phase 2: stability metrics on one split. Zero dollars.

Computes, with family-level bootstrap confidence intervals:
  the transition-status distribution per policy;
  the protocol-conditional supersession rate (the Gate 1 bar);
  the ranking-reversal rate over cutoffs on the multi-protocol subset (the Gate 1 bar);
  the retrospective regret of each policy and the paired differences (the Gate 1 bar);
  conclusion half-life and winner survival curves;
  the protocol-drift rate and the unknown-comparability rate over time.

Writes `checkpoints/metrics_<split>.json`.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replaykg.metrics import (  # noqa: E402
    ORACLES, bootstrap_mean, bootstrap_paired_difference, bootstrap_ratio,
    cross_partition_disagreement, family_regret, kaplan_meier, ranking_reversals,
    structural_change_decomposition, supersession_rates, winner_spells,
)
from replaykg.noise_adapter import noise_model_provenance  # noqa: E402
from replaykg.paths import CHECKPOINTS, SPLIT_MAP  # noqa: E402
from replaykg.replay import POLICIES, replay_family  # noqa: E402
from replaykg.substrate import load_families, verify_substrate  # noqa: E402
from replaykg.transitions import STATUSES, family_transitions  # noqa: E402

PRIMARY_POLICY = "protocol_partitioned"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    args = ap.parse_args()

    digest = verify_substrate()
    assignment = json.loads(SPLIT_MAP.read_text(encoding="utf-8"))["assignment"]
    wanted = {k for k, v in assignment.items() if args.split == "all" or v == args.split}
    families = sorted((f for f in load_families() if f.leaderboard_id in wanted),
                      key=lambda f: f.leaderboard_id)

    replayed = {f.leaderboard_id: replay_family(f) for f in families}
    multi = {f.leaderboard_id for f in families if f.is_multi_protocol}

    direction_unknown = [f.leaderboard_id for f in families
                         if f.metric_direction not in ("higher", "lower")]

    out: dict = {
        "split": args.split,
        "n_families": len(families),
        "n_multi_protocol": len(multi),
        "families_with_unknown_metric_direction": {
            "n": len(direction_unknown),
            "family_ids": sorted(direction_unknown),
            "note": ("an argmax is undefined without a metric direction, so these families "
                     "yield no declared winner at any cutoff and contribute only unresolved "
                     "transitions; they are excluded from every winner-based and regret "
                     "quantity by construction"),
        },
        "substrate_sha256": digest,
        "noise_model": noise_model_provenance(),
        "spend_usd": 0.0,
    }

    # --- transition-status distribution and supersession -------------------------------
    per_policy = {}
    for policy in POLICIES:
        counts_by_family = []
        pooled = collections.Counter()
        for fid, r in replayed.items():
            c = collections.Counter(t.status for t in family_transitions(r, policy))
            counts_by_family.append(dict(c))
            pooled.update(c)
        total = sum(pooled.values())
        shares = {}
        for s in STATUSES:
            per_fam = [(cf.get(s, 0), sum(cf.values())) for cf in counts_by_family]
            shares[s] = {"count": pooled.get(s, 0),
                         "share": (pooled.get(s, 0) / total) if total else None,
                         **bootstrap_ratio(per_fam).as_dict()}
        per_policy[policy] = {
            "n_transitions": total,
            "status_counts": dict(pooled),
            "status_shares": shares,
            "supersession": supersession_rates(counts_by_family),
            "structural_change": structural_change_decomposition(counts_by_family),
        }
    out["transitions"] = per_policy

    # multi-protocol subset, primary policy only
    mp_counts = [dict(collections.Counter(t.status for t in family_transitions(r, PRIMARY_POLICY)))
                 for fid, r in replayed.items() if fid in multi]
    out["supersession_multi_protocol_subset"] = supersession_rates(mp_counts)
    out["structural_change_multi_protocol_subset"] = structural_change_decomposition(mp_counts)
    single_counts = [dict(collections.Counter(
        t.status for t in family_transitions(r, PRIMARY_POLICY)))
        for fid, r in replayed.items() if fid not in multi]
    out["structural_change_single_protocol_subset"] = structural_change_decomposition(
        single_counts)

    # --- retrospective regret, dual oracle ---------------------------------------------
    out["regret"] = {
        "definition": (
            "regret at a cutoff is the direction-aware gap between the best value in the "
            "final-cutoff reference set and the declared method's best value in that same "
            "set, normalised by the reference spread; a declared method absent from the "
            "reference set takes the maximum in-set gap and is counted in the off-partition "
            "rate"
        ),
        "oracles": {
            "protocol_gated": "the largest valid comparison set at the final cutoff",
            "ungated_naive": ("every entry at the final cutoff, protocol ignored; naive "
                              "latest-max has regret zero against this oracle at the final "
                              "cutoff by construction"),
        },
        "by_oracle": {},
    }
    for oracle in ORACLES:
        rows = {p: {} for p in POLICIES}
        for fid, r in replayed.items():
            for p in POLICIES:
                fr = family_regret(r, p, oracle)
                if fr is not None:
                    rows[p][fid] = fr
        scored = sorted(set.intersection(*[set(rows[p]) for p in POLICIES])) if families else []
        out["regret"]["by_oracle"][oracle] = {
            "n_families_scored": len(scored),
            "n_families_excluded_no_reference": len(families) - len(scored),
            "by_policy": {
                p: {
                    "mean_fractional_regret": bootstrap_mean(
                        [rows[p][f]["mean_fractional_regret"] for f in scored]).as_dict(),
                    "mean_regret_points": bootstrap_mean(
                        [rows[p][f]["mean_regret_points"] for f in scored]).as_dict(),
                    "off_partition_rate": bootstrap_mean(
                        [rows[p][f]["off_partition_rate"] for f in scored]).as_dict(),
                } for p in POLICIES
            },
            "paired_difference_vs_naive": {
                p: bootstrap_paired_difference(
                    [(rows[p][f]["mean_fractional_regret"],
                      rows["naive_latest_max"][f]["mean_fractional_regret"]) for f in scored]
                ).as_dict() for p in POLICIES if p != "naive_latest_max"
            },
            "paired_difference_vs_protocol_partitioned": {
                p: bootstrap_paired_difference(
                    [(rows[p][f]["mean_fractional_regret"],
                      rows[PRIMARY_POLICY][f]["mean_fractional_regret"]) for f in scored]
                ).as_dict() for p in ("uncertainty_aware", "conservative_update")
            },
        }

    # --- ranking reversal ---------------------------------------------------------------
    rev_all, rev_multi, pairs_all, pairs_multi = [], [], [], []
    for fid, r in replayed.items():
        rr = ranking_reversals(r, PRIMARY_POLICY)
        rev_all.append((rr["n_with_reversal"], rr["n_transitions"]))
        pairs_all.append((rr["pairs_discordant"], rr["pairs_total"]))
        if fid in multi:
            rev_multi.append((rr["n_with_reversal"], rr["n_transitions"]))
            pairs_multi.append((rr["pairs_discordant"], rr["pairs_total"]))
    out["ranking_reversal"] = {
        "policy": PRIMARY_POLICY,
        "transition_level_all_families": bootstrap_ratio(rev_all).as_dict(),
        "transition_level_multi_protocol": bootstrap_ratio(rev_multi).as_dict(),
        "pair_level_all_families": bootstrap_ratio(pairs_all).as_dict(),
        "pair_level_multi_protocol": bootstrap_ratio(pairs_multi).as_dict(),
    }

    # --- cross-partition disagreement (protocol drift, static form) ----------------------
    cpd = [(d["pairs_disagreeing"], d["pairs_compared"])
           for d in (cross_partition_disagreement(r) for r in replayed.values())]
    out["cross_partition_disagreement"] = bootstrap_ratio(cpd).as_dict()

    # --- survival of the declared winner --------------------------------------------------
    out["survival"] = {}
    for p in POLICIES:
        spells = [s for r in replayed.values() for s in winner_spells(r, p)]
        km = kaplan_meier(spells)
        out["survival"][p] = {
            "kaplan_meier": km,
            "median_half_life_cutoffs": km["median"],
            "n_spells": km["n"], "n_events": km["n_events"],
            "n_censored": km["n"] - km["n_events"],
            "mean_observed_duration": (
                sum(s["duration"] for s in spells) / len(spells)) if spells else None,
        }

    # --- unknown comparability over time ---------------------------------------------------
    by_year = collections.defaultdict(lambda: [0.0, 0])
    for r in replayed.values():
        for t, ev in r["evidence"].items():
            by_year[t][0] += ev.n_unknown
            by_year[t][1] += ev.n_entries
    out["unknown_comparability_over_time"] = {
        str(y): {"unknown_entries": int(v[0]), "entries": v[1],
                 "unknown_share": (v[0] / v[1]) if v[1] else None}
        for y, v in sorted(by_year.items())
    }
    per_fam_unknown = []
    for r in replayed.values():
        years = r["years"]
        if years:
            ev = r["evidence"][years[-1]]
            per_fam_unknown.append((ev.n_unknown, ev.n_entries))
    out["unknown_comparability_final_cutoff"] = bootstrap_ratio(per_fam_unknown).as_dict()

    # --- protocol drift rate ----------------------------------------------------------------
    drift = []
    for r in replayed.values():
        years = r["years"]
        if len(years) < 2:
            continue
        n_sets = [r["conclusions"][PRIMARY_POLICY][i].n_valid_sets for i in range(len(years))]
        grew = sum(1 for a, b in zip(n_sets, n_sets[1:]) if b > a)
        drift.append((grew, len(years) - 1))
    out["protocol_drift_rate"] = {
        "definition": "share of transitions at which the number of valid comparison sets grew",
        **bootstrap_ratio(drift).as_dict(),
    }

    # --- families with no conclusion at all --------------------------------------------------
    never = [fid for fid, r in replayed.items()
             if not any(c.has_conclusion for c in r["conclusions"][PRIMARY_POLICY])]
    out["families_with_no_justified_conclusion_at_any_cutoff"] = {
        "n": len(never), "share": len(never) / len(families) if families else None,
        "family_ids": sorted(never),
    }

    path = CHECKPOINTS / f"metrics_{args.split}.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

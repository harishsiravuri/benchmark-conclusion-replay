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
    common_declared_cutoffs, cross_partition_disagreement, declared_cutoffs, family_regret,
    kaplan_meier, reference_partition, reversal_opportunities, weakened_runs,
    ranking_reversals, structural_change_decomposition, supersession_rates, winner_spells,
)
from replaykg.noise_adapter import noise_model_provenance  # noqa: E402
from replaykg.paths import CHECKPOINTS, SPLIT_MAP, ensure_dirs  # noqa: E402
from replaykg.replay import ALL_POLICIES, POLICIES, replay_family  # noqa: E402
from replaykg.substrate import load_families, verify_substrate  # noqa: E402
from replaykg.transitions import STATUSES, family_transitions  # noqa: E402

PRIMARY_POLICY = "protocol_partitioned"


def _emit_regret(out: dict, oracle: str, basis: str, rows: dict, n_families: int) -> None:
    """Write one oracle-by-basis regret block.

    `scored` is the intersection over FAMILIES for which every policy produced a score, so
    the per-policy means are over the same families. Whether they are also over the same
    cutoffs is what `basis` selects.
    """
    scored = sorted(set.intersection(*[set(rows[p]) for p in POLICIES])) if rows else []
    block = {
        "basis": basis,
        "n_families_scored": len(scored),
        # NB: this counts families dropped for ANY reason, namely no usable reference and
        # also no declared winner under some policy. It is not a count of missing references.
        "n_families_not_scored": n_families - len(scored),
        "n_cutoffs_scored_total": {
            p: sum(rows[p][f]["n_cutoffs_scored"] for f in scored) for p in POLICIES},
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
    out["regret"]["by_oracle"].setdefault(oracle, {})[basis] = block


def main() -> int:
    ensure_dirs()
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
        "bases": {
            "same_cutoff": ("each policy is averaged only over the cutoffs at which EVERY "
                            "policy declared a winner, so the means cover identical cutoffs; "
                            "this is the headline basis"),
            "own_cutoffs": ("each policy is averaged over its own declared cutoffs; naive "
                            "latest-max declares at cutoffs with no valid comparison set, "
                            "where the gated policies decline and where naive does worst, so "
                            "this basis is permissive toward gating"),
        },
        "by_oracle": {},
    }
    for oracle in ORACLES:
        for basis in ("same_cutoff", "own_cutoffs"):
            rows = {p: {} for p in POLICIES}
            for fid, r in replayed.items():
                restrict = common_declared_cutoffs(r) if basis == "same_cutoff" else None
                for p in POLICIES:
                    fr = family_regret(r, p, oracle, restrict)
                    if fr is not None:
                        rows[p][fid] = fr
            _emit_regret(out, oracle, basis, rows, len(families))

    # --- the normalizing baseline, reported separately ----------------------------------
    # A fifth rule that reconciles unit scale but still ignores evaluation conditions. It
    # is scored in its own block, against the same-cutoff basis defined by the four, so
    # that adding it cannot move any figure computed over the four.
    out["regret"]["normalizing_baseline"] = {
        "note": ("naive_normalized reconciles a bounded metric reported as both a fraction "
                 "and a percentage, then takes the argmax over every visible entry. It uses "
                 "no comparability information, so the gap between it and protocol "
                 "partitioning isolates the effect of checking conditions from the effect "
                 "of fixing the unit artefact."),
        "by_oracle": {},
    }
    for oracle in ORACLES:
        pair = ("naive_normalized", PRIMARY_POLICY, "naive_latest_max")
        rows = {p: {} for p in pair}
        for fid, r in replayed.items():
            keep = common_declared_cutoffs(r)          # basis from the original four
            for p in pair:
                fr = family_regret(r, p, oracle, keep)
                if fr is not None:
                    rows[p][fid] = fr
        scored = sorted(set.intersection(*[set(rows[p]) for p in pair])) if families else []
        out["regret"]["normalizing_baseline"]["by_oracle"][oracle] = {
            "basis": "same_cutoff",
            "n_families_scored": len(scored),
            "mean_fractional_regret": {
                p: bootstrap_mean([rows[p][f]["mean_fractional_regret"]
                                   for f in scored]).as_dict() for p in pair},
            "off_partition_rate": {
                p: bootstrap_mean([rows[p][f]["off_partition_rate"]
                                   for f in scored]).as_dict() for p in pair},
            "mean_fractional_regret_measured_only": {
                p: bootstrap_mean([rows[p][f]["mean_fractional_regret_measured_only"]
                                   for f in scored
                                   if rows[p][f]["mean_fractional_regret_measured_only"]
                                   is not None]).as_dict() for p in pair},
            "paired_difference_partitioned_minus_normalized": bootstrap_paired_difference(
                [(rows[PRIMARY_POLICY][f]["mean_fractional_regret"],
                  rows["naive_normalized"][f]["mean_fractional_regret"])
                 for f in scored]).as_dict(),
            "paired_difference_normalized_minus_raw": bootstrap_paired_difference(
                [(rows["naive_normalized"][f]["mean_fractional_regret"],
                  rows["naive_latest_max"][f]["mean_fractional_regret"])
                 for f in scored]).as_dict(),
        }

    # --- measured gaps, with the absence penalty removed --------------------------------
    # The penalised mean mixes a distance measured inside the reference set with a fixed
    # penalty for naming a method that is not in it. Scoring only the cutoffs where the
    # named method IS in the reference set answers a different and narrower question: when
    # a rule names a method we can place, how good is that method?
    POL5 = ("naive_latest_max", "naive_normalized", PRIMARY_POLICY,
            "uncertainty_aware", "conservative_update")
    meas = {p: {} for p in POL5}
    for fid, r in replayed.items():
        keep = common_declared_cutoffs(r)
        for p in POL5:
            fr = family_regret(r, p, "protocol_gated", keep)
            if fr and fr["mean_fractional_regret_measured_only"] is not None:
                meas[p][fid] = fr["mean_fractional_regret_measured_only"]
    common = sorted(set.intersection(*[set(meas[p]) for p in POL5])) if families else []
    out["regret"]["measured_gaps_only"] = {
        "note": ("same-cutoff basis, protocol-gated reference, restricted to cutoffs where "
                 "the named method appears in the reference set; the absence penalty is "
                 "excluded rather than assigned"),
        "n_families_scored": len(common),
        "mean_fractional_regret": {
            p: bootstrap_mean([meas[p][f] for f in common]).as_dict() for p in POL5},
        "paired_difference": {
            f"{a}_minus_{b}": bootstrap_paired_difference(
                [(meas[a][f], meas[b][f]) for f in common]).as_dict()
            for a, b in ((PRIMARY_POLICY, "naive_latest_max"),
                         (PRIMARY_POLICY, "naive_normalized"),
                         ("naive_normalized", "naive_latest_max"),
                         ("uncertainty_aware", PRIMARY_POLICY),
                         ("conservative_update", PRIMARY_POLICY))},
    }

    # --- repeated states, and the opportunity denominator for a reversal ----------------
    runs = [weakened_runs(r, PRIMARY_POLICY) for r in replayed.values()]
    opps = [reversal_opportunities(r, PRIMARY_POLICY) for r in replayed.values()]
    all_runs = [n for w in runs for n in w["run_lengths"]]
    out["repeated_states"] = {
        "note": ("a family whose leading margin sits inside the band stays there until "
                 "something changes, so every cutoff in that stretch is another weakened "
                 "transition; first entries count distinct episodes"),
        "policy": PRIMARY_POLICY,
        "weakened_first_entry": sum(w["first_entry"] for w in runs),
        "weakened_continuation": sum(w["continuation"] for w in runs),
        "weakened_exit_to_separated": sum(w["exit_to_separated"] for w in runs),
        "episode_count": len(all_runs),
        "episode_length_median": (sorted(all_runs)[len(all_runs)//2] if all_runs else None),
        "episode_length_max": (max(all_runs) if all_runs else None),
    }
    out["reversal_opportunities"] = {
        "note": ("an opportunity is a method already in the comparison set when the leader "
                 "was declared that later receives a fresh measurement in that same set, so "
                 "the recorded order between the two could have changed"),
        "policy": PRIMARY_POLICY,
        "opportunities": sum(o["opportunities"] for o in opps),
        "reversed": sum(o["reversed"] for o in opps),
        "transitions_with_opportunity": sum(o["transitions_with_opportunity"] for o in opps),
        "families_with_opportunity": sum(1 for o in opps if o["opportunities"]),
        "unassessable_leader_absent": sum(o["unassessable_leader_absent"] for o in opps),
        "unassessable_set_changed": sum(o["unassessable_set_changed"] for o in opps),
        "reversals": [d for o in opps for d in o["reversals"]],
    }

    # --- why the two bases differ -------------------------------------------------------
    # The cutoffs naive latest-max is scored on but the gated policies are not. These are
    # exactly the cutoffs with no valid comparison set, so they are the whole gap between
    # the two bases and they deserve to be quantified rather than asserted.
    shared_means, extra_means, shared_off, extra_off = [], [], [], []
    n_shared = n_extra = n_fam_extra = 0
    for fid, r in replayed.items():
        ref = reference_partition(r, "protocol_gated")
        if ref is None:
            continue
        keep = common_declared_cutoffs(r)
        sh = family_regret(r, "naive_latest_max", "protocol_gated", keep)
        own = declared_cutoffs(r, "naive_latest_max")
        ex = own - keep
        if sh is not None:
            shared_means.append(sh["mean_fractional_regret"])
            shared_off.append(sh["off_partition_rate"])
            n_shared += sh["n_cutoffs_scored"]
        if ex:
            exr = family_regret(r, "naive_latest_max", "protocol_gated", ex)
            if exr is not None:
                extra_means.append(exr["mean_fractional_regret"])
                extra_off.append(exr["off_partition_rate"])
                n_extra += exr["n_cutoffs_scored"]
                n_fam_extra += 1
    out["regret"]["basis_diagnostic"] = {
        "note": ("naive latest-max declares a winner at cutoffs where no valid comparison "
                 "set exists and the gated policies decline; these are the cutoffs present "
                 "in the own-cutoffs basis and absent from the same-cutoff basis"),
        "oracle": "protocol_gated",
        "n_cutoffs_shared": n_shared,
        "n_cutoffs_naive_only": n_extra,
        "n_families_with_naive_only_cutoffs": n_fam_extra,
        "naive_regret_on_shared_cutoffs": bootstrap_mean(shared_means).as_dict(),
        "naive_regret_on_naive_only_cutoffs": bootstrap_mean(extra_means).as_dict(),
        "naive_off_partition_on_shared_cutoffs": bootstrap_mean(shared_off).as_dict(),
        "naive_off_partition_on_naive_only_cutoffs": bootstrap_mean(extra_off).as_dict(),
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

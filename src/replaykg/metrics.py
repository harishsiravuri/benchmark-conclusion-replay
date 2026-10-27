"""Stability metrics: retrospective regret, survival, ranking reversal, and drift rates.

All confidence intervals are bootstrap percentile intervals over FAMILIES, resampled with
replacement, because transitions within a family are dependent and the family is the
independent unit of the design. The replicate count and the seed are fixed here.

Retrospective regret is defined against the family's final-cutoff reference partition, which
is the largest valid comparison set once every entry is visible. For a policy that declares
method m at cutoff t:

  on-partition   m appears in the reference partition; regret is the direction-aware gap
                 between the reference best value and m's best value in that partition;
  off-partition  m does not appear in the reference partition, so the declaration cannot be
                 justified there; regret takes the maximum in-partition value, that is the
                 gap between the reference best and the reference worst.

The off-partition penalty is a stated modelling choice, it is bounded, and the off-partition
rate is reported beside the regret so a reader can separate the two effects. Regret is
reported in two units: FRACTIONAL regret in the interval 0 to 1, normalised by the reference
spread, which is the primary cross-family quantity because it is scale free; and NORMALISED
POINTS on a 0 to 100 scale, which is comparable to the the upstream resource noise band.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from replaykg.replay import (
    POLICIES, CutoffEvidence, best_entry, best_value_for_method, principal_cluster,
)
from replaykg.transitions import CHANGE_STATUSES, PROTOCOL_CONDITIONAL_STATUSES

BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 13
CI_ALPHA = 0.05


# ---------------------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------------------

@dataclass
class Estimate:
    point: float | None
    lo: float | None
    hi: float | None
    n_units: int
    n_observations: int
    label: str = ""

    def as_dict(self) -> dict:
        return {"point": self.point, "ci95_lo": self.lo, "ci95_hi": self.hi,
                "n_families": self.n_units, "n_observations": self.n_observations,
                "test": "family-level nonparametric bootstrap, 10000 replicates, seed 13"}


def bootstrap_ratio(per_family: list[tuple[float, float]],
                    replicates: int = BOOTSTRAP_REPLICATES,
                    seed: int = BOOTSTRAP_SEED) -> Estimate:
    """Percentile interval for a pooled ratio sum(num) / sum(den) over families."""
    units = [(n, d) for (n, d) in per_family if d > 0]
    n_obs = int(sum(d for _, d in units))
    if not units:
        return Estimate(None, None, None, 0, 0)
    point = sum(n for n, _ in units) / sum(d for _, d in units)
    rng = random.Random(seed)
    k = len(units)
    draws = []
    for _ in range(replicates):
        num = den = 0.0
        for _ in range(k):
            n, d = units[rng.randrange(k)]
            num += n
            den += d
        if den > 0:
            draws.append(num / den)
    draws.sort()
    lo = draws[int(CI_ALPHA / 2 * len(draws))]
    hi = draws[min(len(draws) - 1, int((1 - CI_ALPHA / 2) * len(draws)))]
    return Estimate(point, lo, hi, k, n_obs)


def bootstrap_mean(per_family: list[float],
                   replicates: int = BOOTSTRAP_REPLICATES,
                   seed: int = BOOTSTRAP_SEED,
                   n_observations: int | None = None) -> Estimate:
    """Percentile interval for the mean of one value per family."""
    units = [v for v in per_family if v is not None]
    if not units:
        return Estimate(None, None, None, 0, 0)
    point = sum(units) / len(units)
    rng = random.Random(seed)
    k = len(units)
    draws = []
    for _ in range(replicates):
        draws.append(sum(units[rng.randrange(k)] for _ in range(k)) / k)
    draws.sort()
    lo = draws[int(CI_ALPHA / 2 * len(draws))]
    hi = draws[min(len(draws) - 1, int((1 - CI_ALPHA / 2) * len(draws)))]
    return Estimate(point, lo, hi, k, n_observations if n_observations is not None else k)


def bootstrap_paired_difference(per_family: list[tuple[float, float]],
                                replicates: int = BOOTSTRAP_REPLICATES,
                                seed: int = BOOTSTRAP_SEED) -> Estimate:
    """Percentile interval for the mean paired difference a minus b, paired within family."""
    units = [(a, b) for (a, b) in per_family if a is not None and b is not None]
    if not units:
        return Estimate(None, None, None, 0, 0)
    diffs = [a - b for a, b in units]
    point = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    k = len(diffs)
    draws = []
    for _ in range(replicates):
        draws.append(sum(diffs[rng.randrange(k)] for _ in range(k)) / k)
    draws.sort()
    lo = draws[int(CI_ALPHA / 2 * len(draws))]
    hi = draws[min(len(draws) - 1, int((1 - CI_ALPHA / 2) * len(draws)))]
    return Estimate(point, lo, hi, k, k)


# ---------------------------------------------------------------------------------------
# Reference partition and regret
# ---------------------------------------------------------------------------------------

ORACLES = ("protocol_gated", "ungated_naive")


def _reference(entries, direction, metric, cluster_id):
    best = best_entry(entries, direction)
    worst = best_entry(entries, "lower" if direction == "higher" else "higher")
    if best is None or worst is None or best.value == worst.value:
        return None
    return {"cluster_id": cluster_id, "entries": entries, "best": best, "worst": worst,
            "direction": direction, "metric": metric,
            "spread": abs(best.value - worst.value)}


def reference_partition(replayed: dict, oracle: str = "protocol_gated"):
    """The final-cutoff reference against which retrospective regret is measured.

    Two oracles are supported and both are reported, because they answer different questions.

      protocol_gated  the largest valid comparison set once every entry is visible. This asks
                      how close a policy came to the protocol-justified conclusion.
      ungated_naive   every entry visible at the final cutoff, protocol ignored. This asks how
                      close a policy came to the best reported number regardless of whether
                      that number is comparable to anything. The naive latest-max policy has
                      regret zero against this oracle at the final cutoff by construction, so
                      it is the oracle that is structurally unfavourable to protocol gating,
                      which is precisely why it is worth reporting.
    """
    years = replayed["years"]
    if not years:
        return None
    ev: CutoffEvidence = replayed["evidence"][years[-1]]
    if ev.metric_direction not in ("higher", "lower"):
        return None
    if oracle == "ungated_naive":
        return _reference(ev.entries, ev.metric_direction, ev.metric, None)
    cid = principal_cluster(ev)
    if cid is None:
        return None
    return _reference(ev.clusters[cid], ev.metric_direction, ev.metric, cid)


def _to_points(v: float, reference_max: float) -> float:
    """upstream convention: values on a 0 to 1 scale are read as a 0 to 100 point scale."""
    return v * 100.0 if reference_max <= 1.5 else v


def declared_cutoffs(replayed: dict, policy: str) -> frozenset:
    """Cutoffs at which this policy declared a winner, so could be scored."""
    return frozenset(c.cutoff for c in replayed["conclusions"][policy]
                     if c.winner_method is not None)


def common_declared_cutoffs(replayed: dict, policies=POLICIES) -> frozenset:
    """Cutoffs at which EVERY policy declared a winner.

    This is the same-cutoff comparison basis. The own-cutoffs basis averages each policy
    over its own declarations, which is not like for like: naive latest-max declares a
    winner at every cutoff, including those with no valid comparison set, where the three
    protocol-gated policies correctly decline. Those extra cutoffs are the ones on which
    naive does worst, so scoring it there and its competitors not inflates its regret and
    flatters gating. Both bases are reported; neither is left implicit.
    """
    return frozenset.intersection(*[declared_cutoffs(replayed, p) for p in policies])


def family_regret(replayed: dict, policy: str, oracle: str = "protocol_gated",
                  restrict_to: frozenset | None = None) -> dict | None:
    """Mean regret over the family's cutoffs for one policy, in both units.

    `restrict_to` limits scoring to a given set of cutoffs, which is how the same-cutoff
    basis is computed. None scores every cutoff at which this policy declared a winner.
    """
    ref = reference_partition(replayed, oracle)
    if ref is None:
        return None
    entries, direction = ref["entries"], ref["direction"]
    best_val, spread = ref["best"].value, ref["spread"]
    scale_ref = max(abs(ref["best"].value), abs(ref["worst"].value))

    frac, pts = [], []
    off_partition = no_conclusion = skipped = 0
    for c in replayed["conclusions"][policy]:
        if c.winner_method is None:
            no_conclusion += 1
            continue
        if restrict_to is not None and c.cutoff not in restrict_to:
            skipped += 1
            continue
        v = best_value_for_method(entries, c.winner_method, direction)
        if v is None:
            off_partition += 1
            gap = spread
        else:
            gap = abs(best_val - v)
        frac.append(gap / spread)
        pts.append(_to_points(gap, scale_ref))
    n = len(frac)
    if n == 0:
        return None
    return {
        "family_id": replayed["family"].leaderboard_id,
        "policy": policy,
        "oracle": oracle,
        "n_cutoffs_scored": n,
        "mean_fractional_regret": sum(frac) / n,
        "mean_regret_points": sum(pts) / n,
        "off_partition_rate": off_partition / n,
        "no_conclusion_cutoffs": no_conclusion,
        "cutoffs_skipped_by_restriction": skipped,
        "basis": "same_cutoff" if restrict_to is not None else "own_cutoffs",
        "reference_spread": spread,
    }


# ---------------------------------------------------------------------------------------
# Survival of the declared winner
# ---------------------------------------------------------------------------------------

def winner_spells(replayed: dict, policy: str) -> list[dict]:
    """Spells of an unchanged declared winner, right-censored at the final cutoff.

    A spell starts when a winner is first declared and ends when the declared winner changes
    (an event) or when the family runs out of cutoffs (censored). Cutoffs with no conclusion
    end the current spell as censored, because the conclusion was not overturned, it became
    unavailable.
    """
    concl = replayed["conclusions"][policy]
    spells, cur, length = [], None, 0
    for c in concl:
        if c.winner_method is None:
            if cur is not None:
                spells.append({"winner": cur, "duration": length, "event": 0})
                cur, length = None, 0
            continue
        if cur is None:
            cur, length = c.winner_method, 1
        elif c.winner_method == cur:
            length += 1
        else:
            spells.append({"winner": cur, "duration": length, "event": 1})
            cur, length = c.winner_method, 1
    if cur is not None:
        spells.append({"winner": cur, "duration": length, "event": 0})
    for s in spells:
        s["family_id"] = replayed["family"].leaderboard_id
        s["policy"] = policy
    return spells


def kaplan_meier(spells: list[dict]) -> dict:
    """Kaplan-Meier survival of the declared winner, in cutoffs since declaration."""
    if not spells:
        return {"times": [], "survival": [], "median": None, "n": 0, "n_events": 0}
    times = sorted({s["duration"] for s in spells})
    at_risk = len(spells)
    surv, out_t, out_s = 1.0, [0], [1.0]
    for t in times:
        events = sum(1 for s in spells if s["duration"] == t and s["event"] == 1)
        censored = sum(1 for s in spells if s["duration"] == t and s["event"] == 0)
        if at_risk > 0 and events > 0:
            surv *= (1 - events / at_risk)
        out_t.append(t)
        out_s.append(surv)
        at_risk -= (events + censored)
    median = None
    for t, s in zip(out_t, out_s):
        if s <= 0.5:
            median = t
            break
    return {"times": out_t, "survival": out_s, "median": median,
            "n": len(spells), "n_events": sum(s["event"] for s in spells)}


# ---------------------------------------------------------------------------------------
# Ranking reversal and protocol drift
# ---------------------------------------------------------------------------------------

def ranking_reversals(replayed: dict, policy: str = "protocol_partitioned") -> dict:
    """Discordant-pair reversals in the principal partition between consecutive cutoffs."""
    years, evidence = replayed["years"], replayed["evidence"]
    concl = replayed["conclusions"][policy]
    n_transitions = n_reversed = 0
    pairs_total = pairs_discordant = 0
    for i in range(1, len(years)):
        a, b = concl[i - 1], concl[i]
        if a.winner_cluster is None or b.winner_cluster is None:
            continue
        ev_a, ev_b = evidence[years[i - 1]], evidence[years[i]]
        direction = ev_b.metric_direction
        if direction not in ("higher", "lower"):
            continue
        ma = {e.method: best_value_for_method(ev_a.clusters[a.winner_cluster], e.method, direction)
              for e in ev_a.clusters[a.winner_cluster]}
        mb = {e.method: best_value_for_method(ev_b.clusters[b.winner_cluster], e.method, direction)
              for e in ev_b.clusters[b.winner_cluster]}
        common = sorted(set(ma) & set(mb))
        if len(common) < 2:
            continue
        n_transitions += 1
        disc = 0
        for x in range(len(common)):
            for y in range(x + 1, len(common)):
                p, q = common[x], common[y]
                pairs_total += 1
                if (ma[p] - ma[q]) * (mb[p] - mb[q]) < 0:
                    disc += 1
                    pairs_discordant += 1
        if disc > 0:
            n_reversed += 1
    return {"n_transitions": n_transitions, "n_with_reversal": n_reversed,
            "pairs_total": pairs_total, "pairs_discordant": pairs_discordant}


def cross_partition_disagreement(replayed: dict) -> dict:
    """Method pairs ordered differently in two valid comparison sets at the same cutoff."""
    total = disagreeing = 0
    for t in replayed["years"]:
        ev = replayed["evidence"][t]
        direction = ev.metric_direction
        if direction not in ("higher", "lower"):
            continue
        sets = list(ev.valid_sets.items())
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                ea, eb = sets[i][1], sets[j][1]
                shared = sorted({e.method for e in ea} & {e.method for e in eb})
                for x in range(len(shared)):
                    for y in range(x + 1, len(shared)):
                        p, q = shared[x], shared[y]
                        pa = best_value_for_method(ea, p, direction) - best_value_for_method(ea, q, direction)
                        pb = best_value_for_method(eb, p, direction) - best_value_for_method(eb, q, direction)
                        total += 1
                        if pa * pb < 0:
                            disagreeing += 1
    return {"pairs_compared": total, "pairs_disagreeing": disagreeing}


# ---------------------------------------------------------------------------------------
# Aggregation over families
# ---------------------------------------------------------------------------------------

def supersession_rates(per_family_counts: list[dict]) -> dict:
    """Protocol-attributable share of structural change, with both denominators reported."""
    change = [(sum(c.get(s, 0) for s in PROTOCOL_CONDITIONAL_STATUSES),
               sum(c.get(s, 0) for s in CHANGE_STATUSES)) for c in per_family_counts]
    allt = [(sum(c.get(s, 0) for s in PROTOCOL_CONDITIONAL_STATUSES),
             sum(c.values())) for c in per_family_counts]
    return {
        "among_changes": bootstrap_ratio(change).as_dict(),
        "among_all_transitions": bootstrap_ratio(allt).as_dict(),
    }


# The reframed primary quantity. A transition is UNSETTLED when the justified conclusion did
# not simply carry forward intact, that is when it fragmented, moved with the partition,
# reversed within a protocol, lost its margin, or was cleanly superseded. The load-bearing
# number is the share of unsettled transitions that are NOT a clean same-protocol advance.
UNSETTLED_STATUSES = CHANGE_STATUSES + ("weakened",)
NOT_CLEAN_ADVANCE = ("conditionally_superseded", "fragmented_by_protocol_drift",
                     "contradicted", "weakened")


def structural_change_decomposition(per_family_counts: list[dict]) -> dict:
    """Protocol-driven structural-change rate and its full decomposition.

    Reported instead of a headline supersession rate, because fragmentation dominates the
    protocol-attributable mass and fragmentation means no single justified winner remains,
    which is a different claim from one winner replacing another.
    """
    def ratio(numer_statuses, denom_statuses):
        return bootstrap_ratio([
            (sum(c.get(s, 0) for s in numer_statuses),
             sum(c.get(s, 0) for s in denom_statuses)) for c in per_family_counts]).as_dict()

    pooled = {}
    for c in per_family_counts:
        for s, n in c.items():
            pooled[s] = pooled.get(s, 0) + n
    n_unsettled = sum(pooled.get(s, 0) for s in UNSETTLED_STATUSES)
    n_resolved = sum(pooled.get(s, 0) for s in UNSETTLED_STATUSES) + pooled.get(
        "remains_supported", 0)

    return {
        "definition": {
            "unsettled": list(UNSETTLED_STATUSES),
            "not_clean_advance": list(NOT_CLEAN_ADVANCE),
            "resolved": list(UNSETTLED_STATUSES) + ["remains_supported"],
        },
        "counts": {s: pooled.get(s, 0) for s in
                   list(UNSETTLED_STATUSES) + ["remains_supported", "unresolved", "established"]},
        "n_unsettled": n_unsettled,
        "n_resolved": n_resolved,
        # headline: of transitions where the conclusion did not simply carry forward, what
        # share is something other than a clean same-protocol advance
        "not_clean_advance_among_unsettled": ratio(NOT_CLEAN_ADVANCE, UNSETTLED_STATUSES),
        # headline: of all transitions with a justified conclusion at both ends, what share is
        # unsettled at all
        "unsettled_among_resolved": ratio(UNSETTLED_STATUSES,
                                          UNSETTLED_STATUSES + ("remains_supported",)),
        # decomposition, each against the unsettled denominator
        "decomposition_among_unsettled": {
            s: ratio((s,), UNSETTLED_STATUSES) for s in UNSETTLED_STATUSES
        },
        # the strict protocol-attributable measure, kept for continuity with the Gate 1 report
        "protocol_attributable_among_changes": ratio(PROTOCOL_CONDITIONAL_STATUSES,
                                                     CHANGE_STATUSES),
        "protocol_attributable_among_unsettled": ratio(PROTOCOL_CONDITIONAL_STATUSES,
                                                       UNSETTLED_STATUSES),
    }

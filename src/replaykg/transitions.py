"""Classification of cutoff-to-cutoff transitions in the declared conclusion.

The seven statuses the chapter protocol fixes are:

  remains_supported            the declared winner stands and its lead is beyond the band;
  superseded_same_protocol     a method not previously in the comparison set arrives and wins,
                               sharing a comparison set with the method it replaces;
  conditionally_superseded     the selected method changes because the largest comparison
                               set is now a different one, including the case where the
                               previously selected method has no result in the new set;
  contradicted                 a within-protocol reversal beyond the band among methods that
                               were already in the comparison set;
  fragmented_by_protocol_drift the comparable partition splits and the argmax answer stops
                               being robust across the valid comparison sets;
  weakened                     the lead falls inside the noise band, so no successor is clear;
  unresolved                   unknown comparability dominates, or no comparison set exists.

One bookkeeping status is added, because the seven do not cover the case where a family holds
no conclusion at t minus one and acquires one at t:

  established                  the first justified conclusion after a cutoff with none.

`established` is reported separately and is excluded from the supersession-rate denominator.

The classifier is a deterministic ordered decision procedure. The order matters and is fixed
here, so every transition receives exactly one status.
"""

from __future__ import annotations

from dataclasses import dataclass

from replaykg.noise_adapter import beyond_noise
from replaykg.replay import UNRESOLVED_UNKNOWN_SHARE, Conclusion, CutoffEvidence, best_value_for_method

STATUSES = (
    "remains_supported",
    "superseded_same_protocol",
    "conditionally_superseded",
    "contradicted",
    "fragmented_by_protocol_drift",
    "weakened",
    "unresolved",
    "established",
)

# The transitions that represent a structural change in the conclusion. This is the
# denominator of the protocol-conditional supersession rate.
CHANGE_STATUSES = (
    "superseded_same_protocol",
    "conditionally_superseded",
    "contradicted",
    "fragmented_by_protocol_drift",
)

# The numerator: change that is attributable to protocol heterogeneity rather than to clean
# within-protocol progress.
PROTOCOL_CONDITIONAL_STATUSES = (
    "conditionally_superseded",
    "fragmented_by_protocol_drift",
)


@dataclass
class Transition:
    family_id: str
    policy: str
    from_cutoff: int
    to_cutoff: int
    status: str
    from_winner: str | None
    to_winner: str | None
    from_cluster: int | None
    to_cluster: int | None
    reason: str
    evidence_size: int
    unknown_share: float
    n_valid_sets_from: int
    n_valid_sets_to: int

    def as_dict(self) -> dict:
        return {
            "family_id": self.family_id, "policy": self.policy,
            "from_cutoff": self.from_cutoff, "to_cutoff": self.to_cutoff,
            "status": self.status, "reason": self.reason,
            "from_winner": self.from_winner, "to_winner": self.to_winner,
            "from_cluster": self.from_cluster, "to_cluster": self.to_cluster,
            "evidence_size": self.evidence_size, "unknown_share": self.unknown_share,
            "n_valid_sets_from": self.n_valid_sets_from, "n_valid_sets_to": self.n_valid_sets_to,
        }


def _methods_in_set(ev: CutoffEvidence, cluster_id: int | None) -> set[str]:
    if cluster_id is None:
        return {e.method for e in ev.entries}
    return {e.method for e in ev.clusters.get(cluster_id, ())}


def _first_year_in_set(ev: CutoffEvidence, cluster_id: int | None, method: str) -> int | None:
    """Earliest year at which a method became available inside the given comparison set."""
    pool = ev.entries if cluster_id is None else ev.clusters.get(cluster_id, ())
    years = [e.year for e in pool if e.method == method and e.year is not None]
    return min(years) if years else None


def classify(prev: Conclusion, cur: Conclusion,
             ev_prev: CutoffEvidence, ev_cur: CutoffEvidence,
             unresolved_threshold: float = UNRESOLVED_UNKNOWN_SHARE,
             band: float = 1.0) -> Transition:
    """Ordered decision procedure. Exactly one status is returned."""
    common = dict(
        family_id=cur.family_id, policy=cur.policy,
        from_cutoff=prev.cutoff, to_cutoff=cur.cutoff,
        from_winner=prev.winner_method, to_winner=cur.winner_method,
        from_cluster=prev.winner_cluster, to_cluster=cur.winner_cluster,
        evidence_size=cur.evidence_size, unknown_share=cur.unknown_share,
        n_valid_sets_from=prev.n_valid_sets, n_valid_sets_to=cur.n_valid_sets,
    )

    # 1. unresolved: no conclusion is available, or the unknown mass dominates.
    if not cur.has_conclusion:
        return Transition(status="unresolved",
                          reason="no justified conclusion at the later cutoff", **common)
    if cur.unknown_share > unresolved_threshold and cur.policy != "naive_latest_max":
        return Transition(
            status="unresolved",
            reason=f"unknown-comparability share {cur.unknown_share:.2f} exceeds "
                   f"{unresolved_threshold:.2f}", **common)

    # 2. establishment out of a cutoff that carried no conclusion.
    if not prev.has_conclusion:
        return Transition(status="established",
                          reason="first justified conclusion after a cutoff with none", **common)

    # 3. fragmentation: the answer stops being robust as the partition splits.
    if (prev.answer_status == "robust" and cur.answer_status == "conditional"
            and cur.n_valid_sets > prev.n_valid_sets):
        return Transition(
            status="fragmented_by_protocol_drift",
            reason=f"valid comparison sets grew from {prev.n_valid_sets} to "
                   f"{cur.n_valid_sets} and the argmax answer is no longer robust", **common)

    # 4. the declared winner is unchanged.
    if cur.winner_method == prev.winner_method:
        if cur.margin_beyond_noise is False:
            return Transition(
                status="weakened",
                reason="the declared winner stands but its margin is inside the noise band",
                **common)
        return Transition(status="remains_supported",
                          reason="the declared winner stands beyond the noise band", **common)

    # 5. the declared winner changed.
    #    5a. a flip inside the noise band is not a clear succession.
    if cur.winner_cluster is not None:
        prior_val = best_value_for_method(
            ev_cur.clusters.get(cur.winner_cluster, ()), prev.winner_method,
            ev_cur.metric_direction)
        if prior_val is not None and not beyond_noise(
                cur.winner_value, prior_val, ev_cur.metric, band):
            return Transition(
                status="weakened",
                reason="the winner flipped but the new lead is inside the noise band",
                **common)

    #    5b. the largest comparison set is now a different one. The proposal routes every
    #        such change here, including the case where the previously selected method has
    #        no result in the new set, because the two winners never shared a set and the
    #        change therefore says nothing about overtaking under matching conditions.
    if (prev.winner_cluster is not None and cur.winner_cluster is not None
            and cur.winner_cluster != prev.winner_cluster):
        still = cur.cluster_winners.get(prev.winner_cluster)
        if still is not None and still["method"] == prev.winner_method:
            tail = "the earlier winner still leads its own partition"
        else:
            tail = "the earlier winner does not lead the set it was selected from"
        return Transition(
            status="conditionally_superseded",
            reason=f"the principal partition moved from cluster {prev.winner_cluster} to "
                   f"{cur.winner_cluster}; {tail}",
            **common)

    #    5c. arrival and reversal both require the two winners to share a comparison set.
    #        Without a shared set there is no comparison between them to supersede.
    if cur.winner_cluster is not None:
        shared = best_value_for_method(
            ev_cur.clusters.get(cur.winner_cluster, ()), prev.winner_method,
            ev_cur.metric_direction)
        if shared is None:
            return Transition(
                status="conditionally_superseded",
                reason=f"the previous winner has no result in cluster {cur.winner_cluster}, "
                       f"so the two winners never shared a comparison set", **common)

    #    5d. arrival versus reversal, measured against the cutoff at which the incumbent was
    #        declared rather than against the previous cutoff, so a policy's declaration lag
    #        cannot turn an arrival into an apparent reversal.
    reference = prev.declared_since if prev.declared_since is not None else prev.cutoff
    first_seen = _first_year_in_set(ev_cur, cur.winner_cluster, cur.winner_method)
    if first_seen is not None and first_seen <= reference:
        return Transition(
            status="contradicted",
            reason=f"the new winner was already comparable and losing at cutoff {reference}, "
                   f"so the change is a within-protocol reversal beyond the noise band",
            **common)

    #    5e. otherwise a new method arrived and won within the same comparison set.
    return Transition(
        status="superseded_same_protocol",
        reason=f"a method first comparable at cutoff {first_seen} arrived and won", **common)


def family_transitions(replayed: dict, policy: str,
                       unresolved_threshold: float = UNRESOLVED_UNKNOWN_SHARE,
                       band: float = 1.0) -> list[Transition]:
    years = replayed["years"]
    evidence = replayed["evidence"]
    concl = replayed["conclusions"][policy]
    out = []
    for i in range(1, len(years)):
        out.append(classify(concl[i - 1], concl[i], evidence[years[i - 1]], evidence[years[i]],
                            unresolved_threshold, band))
    return out

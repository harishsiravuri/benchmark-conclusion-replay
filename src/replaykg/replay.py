"""Chronological replay: evidence reconstruction at yearly cutoffs and the four policies.

No future leakage is the governing invariant. At cutoff year t a family sees only entries
whose publication year is at most t. `CutoffEvidence` validates that invariant when it is
constructed, and it is the only object the policies are given, so a policy has no route to a
later entry. `tests/test_leakage.py` additionally checks the behavioural form of the
invariant, namely that perturbing an entry dated after t changes no conclusion at t.

The valid comparison sets are the upstream comparable clusters, consumed frozen. Each
cluster is a maximal set of entries agreeing on the observed must-agree facets split, metric
surface, and unit, which is exactly the query-induced maximal compatible subset of the
upstream framework. Entries missing a must-agree facet are the comparability-unknown mass
and are excluded from every valid comparison set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from replaykg.noise_adapter import beyond_noise, classify_metric_range
from replaykg.substrate import Entry, Family

POLICIES = (
    "naive_latest_max",
    "protocol_partitioned",
    "uncertainty_aware",
    "conservative_update",
)

# A diagnostic baseline, reported beside the four but deliberately NOT a member of
# POLICIES. POLICIES defines the same-cutoff comparison basis, and adding a fifth rule
# there would silently move every published same-cutoff figure.
DIAGNOSTIC_POLICIES = ("naive_normalized",)
ALL_POLICIES = POLICIES + DIAGNOSTIC_POLICIES

# Preregistered policy parameters (see PREREGISTRATION.md).
CONSERVATIVE_K = 2                 # consecutive cutoffs a challenger must lead
UNRESOLVED_UNKNOWN_SHARE = 0.5     # unknown mass above which a cutoff is unresolved
MIN_VALID_SET_SIZE = 2             # a valid comparison set needs at least two entries


class LeakageError(AssertionError):
    """Raised when evidence carrying a year after the cutoff reaches a computation."""


# ---------------------------------------------------------------------------------------
# Evidence reconstruction
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CutoffEvidence:
    """Everything a family may see at cutoff year t, and nothing else."""

    family_id: str
    cutoff: int
    metric: str
    metric_direction: str
    entries: tuple[Entry, ...]
    clusters: dict[int, tuple[Entry, ...]]
    unknown: tuple[Entry, ...]

    def __post_init__(self) -> None:
        for e in self.entries:
            if e.year is None or e.year > self.cutoff:
                raise LeakageError(
                    f"{self.family_id}: entry {e.arxiv_id} dated {e.year} reached "
                    f"cutoff {self.cutoff}"
                )

    @property
    def n_entries(self) -> int:
        return len(self.entries)

    @property
    def n_unknown(self) -> int:
        return len(self.unknown)

    @property
    def unknown_share(self) -> float:
        return (self.n_unknown / self.n_entries) if self.n_entries else 1.0

    @property
    def valid_sets(self) -> dict[int, tuple[Entry, ...]]:
        """Clusters large enough to support a comparison at this cutoff."""
        return {c: es for c, es in self.clusters.items() if len(es) >= MIN_VALID_SET_SIZE}


def cutoff_years(family: Family) -> list[int]:
    """Yearly cutoffs from the family's earliest to its latest dated result, inclusive."""
    ys = family.years
    return list(range(ys[0], ys[-1] + 1)) if ys else []


def evidence_at(family: Family, cutoff: int) -> CutoffEvidence:
    visible = tuple(
        e for e in family.entries
        if e.year is not None and e.year <= cutoff and e.value is not None
    )
    clusters: dict[int, list[Entry]] = {}
    unknown: list[Entry] = []
    for e in visible:
        if e.cluster_id is None:
            unknown.append(e)
        else:
            clusters.setdefault(e.cluster_id, []).append(e)
    return CutoffEvidence(
        family_id=family.leaderboard_id,
        cutoff=cutoff,
        metric=family.metric,
        metric_direction=family.metric_direction,
        entries=visible,
        clusters={c: tuple(sorted(v, key=lambda x: x.entry_index)) for c, v in sorted(clusters.items())},
        unknown=tuple(sorted(unknown, key=lambda x: x.entry_index)),
    )


# ---------------------------------------------------------------------------------------
# Direction-aware comparison
# ---------------------------------------------------------------------------------------

def is_better(a: float, b: float, direction: str) -> bool:
    """True when value a is a better result than value b under the metric direction."""
    if direction == "higher":
        return a > b
    if direction == "lower":
        return a < b
    return False


def best_entry(entries, direction: str) -> Entry | None:
    """Best entry under the metric direction. Ties break on the stable entry index."""
    if direction not in ("higher", "lower"):
        return None
    best = None
    for e in entries:
        if e.value is None:
            continue
        if best is None or is_better(e.value, best.value, direction) or (
            e.value == best.value and e.entry_index < best.entry_index
        ):
            best = e
    return best


def best_value_for_method(entries, method: str, direction: str) -> float | None:
    vals = [e.value for e in entries if e.method == method and e.value is not None]
    if not vals:
        return None
    return max(vals) if direction == "higher" else min(vals)


# ---------------------------------------------------------------------------------------
# Conclusions
# ---------------------------------------------------------------------------------------

@dataclass
class Conclusion:
    """The justified conclusion for one family, one cutoff, one policy."""

    family_id: str
    policy: str
    cutoff: int
    winner_method: str | None
    winner_value: float | None
    winner_cluster: int | None
    principal_signature: dict | None
    evidence_size: int
    comparison_set_size: int
    unknown_size: int
    unknown_share: float
    n_valid_sets: int
    answer_status: str                     # robust | conditional | insufficient | undefined
    margin_over_runner_up: float | None
    margin_beyond_noise: bool | None
    cluster_winners: dict = field(default_factory=dict)   # cluster_id -> {method, value, n}
    shared_methods_across_sets: int = 0
    declared_since: int | None = None    # cutoff at which this winner was first declared
    notes: list = field(default_factory=list)

    @property
    def has_conclusion(self) -> bool:
        return self.winner_method is not None


def _cluster_winners(ev: CutoffEvidence) -> dict:
    out = {}
    for cid, entries in ev.valid_sets.items():
        b = best_entry(entries, ev.metric_direction)
        if b is not None:
            out[cid] = {"method": b.method, "value": b.value, "n": len(entries)}
    return out


def _answer_status(cw: dict, ev: CutoffEvidence) -> str:
    """Upstream framework status of the argmax answer over the valid comparison sets."""
    if ev.metric_direction not in ("higher", "lower"):
        return "undefined"
    if not cw:
        return "insufficient"
    winners = {v["method"] for v in cw.values()}
    if len(cw) == 1:
        return "robust"
    return "robust" if len(winners) == 1 else "conditional"


def _shared_methods(ev: CutoffEvidence) -> int:
    sets = [{e.method for e in es} for es in ev.valid_sets.values()]
    if len(sets) < 2:
        return 0
    shared = set()
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            shared |= sets[i] & sets[j]
    return len(shared)


def _margin(entries, winner: Entry, direction: str, metric: str, band: float = 1.0):
    """Winner's margin over the best other method in the same comparison set."""
    return _margin_for_method(entries, winner.method, winner.value, direction, metric, band)


def _margin_for_method(entries, method: str, value: float, direction: str, metric: str,
                       band: float = 1.0):
    """Margin of a named method, used when a policy holds a winner other than the argmax."""
    others = [e for e in entries if e.method != method and e.value is not None]
    runner = best_entry(others, direction)
    if runner is None:
        return None, None
    return abs(value - runner.value), beyond_noise(value, runner.value, metric, band)


def _base(ev: CutoffEvidence, policy: str) -> dict:
    return {
        "family_id": ev.family_id,
        "policy": policy,
        "cutoff": ev.cutoff,
        "evidence_size": ev.n_entries,
        "unknown_size": ev.n_unknown,
        "unknown_share": ev.unknown_share,
    }


def conclude_naive(ev: CutoffEvidence, band: float = 1.0) -> Conclusion:
    """Naive latest-max: argmax over every entry at or before t, ignoring protocol."""
    w = best_entry(ev.entries, ev.metric_direction)
    margin, bn = (None, None)
    if w is not None:
        margin, bn = _margin(ev.entries, w, ev.metric_direction, ev.metric, band)
    cw = _cluster_winners(ev)
    return Conclusion(
        **_base(ev, "naive_latest_max"),
        winner_method=w.method if w else None,
        winner_value=w.value if w else None,
        winner_cluster=w.cluster_id if w else None,
        principal_signature=dict(w.protocol) if (w and w.protocol) else None,
        comparison_set_size=ev.n_entries,
        n_valid_sets=len(ev.valid_sets),
        answer_status="undefined" if ev.metric_direction not in ("higher", "lower") else "naive",
        margin_over_runner_up=margin,
        margin_beyond_noise=bn,
        cluster_winners=cw,
        shared_methods_across_sets=_shared_methods(ev),
        notes=["protocol ignored by construction"],
    )


def unit_reconciled_values(ev: CutoffEvidence) -> dict:
    """Entry index to a value brought onto one scale, for the normalizing baseline.

    A bounded metric is reported both as a fraction and as a percentage in this archive,
    so 0.8976 and 79.0 can sit on the same board. Comparing them without reconciling the
    scale is a unit error rather than a comparability failure, and a baseline that makes
    that error understates what an ordinary reader of a leaderboard would do.

    The rule is deliberately narrow. For a bounded metric only, when the visible values
    straddle the 1.5 boundary, values at or below 1.5 are read as fractions and multiplied
    by 100. Unbounded metrics are left alone, since no fixed scale relation holds for
    them. Nothing here uses the evaluation conditions, so this baseline remains blind to
    comparability by construction.
    """
    vals = {e.entry_index: e.value for e in ev.entries if e.value is not None}
    if classify_metric_range(ev.metric) != "bounded" or not vals:
        return vals
    low = [v for v in vals.values() if abs(v) <= 1.5]
    high = [v for v in vals.values() if abs(v) > 1.5]
    if not (low and high):
        return vals
    return {i: (v * 100.0 if abs(v) <= 1.5 else v) for i, v in vals.items()}


def conclude_naive_normalized(ev: CutoffEvidence, band: float = 1.0) -> Conclusion:
    """Argmax over every visible entry after reconciling unit scale, protocol ignored.

    This isolates the effect of checking evaluation conditions from the effect of fixing a
    unit artefact. It differs from naive latest-maximum only in the scale reconciliation,
    and from protocol partitioning in using no comparability information at all.
    """
    vals = unit_reconciled_values(ev)
    direction = ev.metric_direction
    best = None
    if direction in ("higher", "lower"):
        for e in ev.entries:
            v = vals.get(e.entry_index)
            if v is None:
                continue
            if best is None:
                best = (e, v)
            elif is_better(v, best[1], direction) or (v == best[1] and e.entry_index < best[0].entry_index):
                best = (e, v)
    margin = bn = None
    if best is not None:
        others = [(e, vals[e.entry_index]) for e in ev.entries
                  if e.method != best[0].method and e.entry_index in vals]
        runner = None
        for e, v in others:
            if runner is None or is_better(v, runner[1], direction):
                runner = (e, v)
        if runner is not None:
            margin = abs(best[1] - runner[1])
            bn = beyond_noise(best[1], runner[1], ev.metric, band)
    return Conclusion(
        **_base(ev, "naive_normalized"),
        winner_method=best[0].method if best else None,
        winner_value=best[1] if best else None,
        winner_cluster=best[0].cluster_id if best else None,
        principal_signature=dict(best[0].protocol) if (best and best[0].protocol) else None,
        comparison_set_size=ev.n_entries,
        n_valid_sets=len(ev.valid_sets),
        answer_status="undefined" if direction not in ("higher", "lower") else "naive",
        margin_over_runner_up=margin,
        margin_beyond_noise=bn,
        cluster_winners=_cluster_winners(ev),
        shared_methods_across_sets=_shared_methods(ev),
        notes=["unit scale reconciled; protocol ignored by construction"],
    )


def principal_cluster(ev: CutoffEvidence) -> int | None:
    """The largest valid comparison set at this cutoff. Ties break on the lowest id."""
    vs = ev.valid_sets
    if not vs:
        return None
    return sorted(vs.items(), key=lambda kv: (-len(kv[1]), kv[0]))[0][0]


def conclude_protocol_partitioned(ev: CutoffEvidence, band: float = 1.0) -> Conclusion:
    """Restrict to the valid comparison set, then take the argmax within it."""
    cid = principal_cluster(ev)
    cw = _cluster_winners(ev)
    status = _answer_status(cw, ev)
    if cid is None:
        return Conclusion(
            **_base(ev, "protocol_partitioned"),
            winner_method=None, winner_value=None, winner_cluster=None,
            principal_signature=None, comparison_set_size=0,
            n_valid_sets=0, answer_status=status,
            margin_over_runner_up=None, margin_beyond_noise=None,
            cluster_winners=cw, shared_methods_across_sets=_shared_methods(ev),
            notes=["no valid comparison set at this cutoff"],
        )
    entries = ev.clusters[cid]
    w = best_entry(entries, ev.metric_direction)
    margin, bn = (None, None) if w is None else _margin(
        entries, w, ev.metric_direction, ev.metric, band)
    return Conclusion(
        **_base(ev, "protocol_partitioned"),
        winner_method=w.method if w else None,
        winner_value=w.value if w else None,
        winner_cluster=cid,
        principal_signature=dict(entries[0].protocol) if entries[0].protocol else None,
        comparison_set_size=len(entries),
        n_valid_sets=len(ev.valid_sets),
        answer_status=status,
        margin_over_runner_up=margin,
        margin_beyond_noise=bn,
        cluster_winners=cw,
        shared_methods_across_sets=_shared_methods(ev),
    )


def conclude_uncertainty_aware(ev: CutoffEvidence, incumbent: str | None,
                               band: float = 1.0) -> Conclusion:
    """Declare a new winner only when the challenger beats the incumbent beyond the band."""
    base = conclude_protocol_partitioned(ev, band)
    base.policy = "uncertainty_aware"
    cid = base.winner_cluster
    if cid is None or base.winner_method is None:
        base.notes.append("no valid comparison set; incumbent retained as pending")
        return base
    entries = ev.clusters[cid]
    challenger = base.winner_method
    if incumbent is None or incumbent == challenger:
        return base
    inc_val = best_value_for_method(entries, incumbent, ev.metric_direction)
    if inc_val is None:
        base.notes.append("incumbent absent from the valid comparison set; challenger declared")
        return base
    if beyond_noise(base.winner_value, inc_val, ev.metric, band):
        base.notes.append("challenger beats the incumbent beyond the noise band")
        return base
    base.winner_method = incumbent
    base.winner_value = inc_val
    base.margin_over_runner_up, base.margin_beyond_noise = _margin_for_method(
        entries, incumbent, inc_val, ev.metric_direction, ev.metric, band)
    base.notes.append("challenger lead is within the noise band; incumbent stands")
    return base


def conclude_conservative(ev: CutoffEvidence, declared: str | None,
                          pending: str | None, pending_count: int,
                          k: int = CONSERVATIVE_K, band: float = 1.0):
    """Require a challenger to lead for k consecutive cutoffs before the winner updates."""
    base = conclude_protocol_partitioned(ev, band)
    base.policy = "conservative_update"
    cid = base.winner_cluster
    leader = base.winner_method
    if cid is None or leader is None:
        base.notes.append("no valid comparison set; declaration withheld")
        return base, declared, pending, pending_count
    entries = ev.clusters[cid]
    if declared is None:
        base.notes.append("first justified conclusion; declared without persistence")
        return base, leader, None, 0
    if leader == declared:
        return base, declared, None, 0
    declared_val = best_value_for_method(entries, declared, ev.metric_direction)
    if declared_val is None:
        base.notes.append("declared winner absent from the valid comparison set; leader declared")
        return base, leader, None, 0
    count = pending_count + 1 if pending == leader else 1
    if count >= k:
        base.notes.append(f"challenger led for {count} consecutive cutoffs; winner updated")
        return base, leader, None, 0
    base.winner_method = declared
    base.winner_value = declared_val
    base.margin_over_runner_up, base.margin_beyond_noise = _margin_for_method(
        entries, declared, declared_val, ev.metric_direction, ev.metric, band)
    base.notes.append(f"challenger has led for {count} of {k} required cutoffs; winner held")
    return base, declared, leader, count


# ---------------------------------------------------------------------------------------
# Full-family replay
# ---------------------------------------------------------------------------------------

def replay_family(family: Family, k: int = CONSERVATIVE_K, band: float = 1.0) -> dict:
    """All four policies over every yearly cutoff for one family.

    `k` and `band` carry their preregistered values in the primary analysis and are varied
    only by the sensitivity script.
    """
    years = cutoff_years(family)
    evidence = {t: evidence_at(family, t) for t in years}
    out: dict[str, list[Conclusion]] = {p: [] for p in ALL_POLICIES}

    for t in years:
        ev = evidence[t]
        out["naive_latest_max"].append(conclude_naive(ev, band))
        out["naive_normalized"].append(conclude_naive_normalized(ev, band))
        out["protocol_partitioned"].append(conclude_protocol_partitioned(ev, band))

    incumbent = None
    for t in years:
        c = conclude_uncertainty_aware(evidence[t], incumbent, band)
        out["uncertainty_aware"].append(c)
        if c.winner_method is not None:
            incumbent = c.winner_method

    declared, pending, count = None, None, 0
    for t in years:
        c, declared, pending, count = conclude_conservative(
            evidence[t], declared, pending, count, k, band)
        out["conservative_update"].append(c)

    for series in out.values():
        _stamp_declared_since(series)

    return {"family": family, "years": years, "evidence": evidence, "conclusions": out}


def _stamp_declared_since(series: list[Conclusion]) -> None:
    """Record, for each conclusion, the cutoff at which its winner was first declared.

    This makes the arrival-versus-reversal test in the transition classifier insensitive to a
    policy's declaration lag: a conservative policy that holds a winner for one extra cutoff
    must not turn an arrival into an apparent reversal.
    """
    since, prev = None, None
    for c in series:
        if c.winner_method is None:
            since, prev = None, None
            continue
        if c.winner_method != prev:
            since = c.cutoff
        c.declared_since = since
        prev = c.winner_method

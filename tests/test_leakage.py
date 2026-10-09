"""No-future-leakage tests. These are the load-bearing tests of the study.

Three independent forms of the invariant are checked:

  structural   `CutoffEvidence` refuses to hold an entry dated after its cutoff;
  filtering    `evidence_at` excludes every entry dated after the cutoff;
  behavioural  perturbing an entry dated after t changes no conclusion at or before t.

The behavioural test is the strongest, because it would fail even if some future path read
the substrate directly rather than going through `evidence_at`.
"""

from __future__ import annotations

import copy
import json

import pytest

from replaykg.paths import SPLIT_MAP
from replaykg.replay import (
    CutoffEvidence, LeakageError, cutoff_years, evidence_at, replay_family,
)
from replaykg.substrate import Entry, load_families


def _dev_families(limit: int = 12):
    assignment = json.loads(SPLIT_MAP.read_text(encoding="utf-8"))["assignment"]
    wanted = {k for k, v in assignment.items() if v == "dev"}
    fams = [f for f in load_families() if f.leaderboard_id in wanted]
    fams.sort(key=lambda f: f.leaderboard_id)
    return fams[:limit]


def test_cutoff_evidence_rejects_a_future_entry():
    future = Entry(method="M", value=1.0, arxiv_id="2401.00001", year=2024,
                   source_location="table:1", cluster_id=0, protocol={}, unknown_reason=None,
                   entry_index=0)
    with pytest.raises(LeakageError):
        CutoffEvidence(family_id="x", cutoff=2020, metric="acc", metric_direction="higher",
                       entries=(future,), clusters={0: (future,)}, unknown=())


def test_evidence_at_filters_every_future_entry():
    for fam in _dev_families():
        for t in cutoff_years(fam):
            ev = evidence_at(fam, t)
            assert all(e.year <= t for e in ev.entries)
            assert all(e.year <= t for es in ev.clusters.values() for e in es)
            assert all(e.year <= t for e in ev.unknown)


def test_evidence_at_is_monotone_in_the_cutoff():
    for fam in _dev_families():
        years = cutoff_years(fam)
        for a, b in zip(years, years[1:]):
            earlier = {e.entry_index for e in evidence_at(fam, a).entries}
            later = {e.entry_index for e in evidence_at(fam, b).entries}
            assert earlier <= later


def _winners(fam):
    replayed = replay_family(fam)
    return {
        (policy, c.cutoff): c.winner_method
        for policy, series in replayed["conclusions"].items()
        for c in series
    }


def test_perturbing_a_future_entry_changes_no_earlier_conclusion():
    """The behavioural form of the invariant.

    For each family, the latest-dated entry's value is replaced by a value far better than
    any other. Every conclusion at a cutoff strictly before that entry's year must be
    unchanged. A leak of any kind would move at least one of them.
    """
    checked = 0
    for fam in _dev_families():
        years = fam.years
        if len(years) < 2:
            continue
        last_year = years[-1]
        baseline = _winners(fam)

        perturbed = copy.deepcopy(fam)
        bumped = False
        for i, e in enumerate(perturbed.entries):
            if e.year == last_year and e.value is not None:
                perturbed.entries[i] = Entry(
                    method="INJECTED_FUTURE_METHOD", value=1e9 if fam.metric_direction != "lower" else -1e9,
                    arxiv_id=e.arxiv_id, year=e.year, source_location=e.source_location,
                    cluster_id=e.cluster_id, protocol=e.protocol,
                    unknown_reason=e.unknown_reason, entry_index=e.entry_index)
                bumped = True
                break
        if not bumped:
            continue

        after = _winners(perturbed)
        for (policy, cutoff), winner in baseline.items():
            if cutoff < last_year:
                assert after[(policy, cutoff)] == winner, (
                    f"{fam.leaderboard_id}: policy {policy} at cutoff {cutoff} moved after a "
                    f"{last_year} entry was perturbed"
                )
        checked += 1
    assert checked >= 5, "the leakage probe exercised too few families to be meaningful"


def test_the_leakage_probe_can_actually_fail():
    """A mutation check: the probe above must detect a genuine leak.

    A deliberately leaky conclusion function that reads every entry regardless of the cutoff
    is confirmed to disagree with the sealed one, which shows the probe has power.
    """
    from replaykg.replay import best_entry

    fam = next(f for f in _dev_families() if len(f.years) >= 3
               and f.metric_direction in ("higher", "lower"))
    early = fam.years[0]
    sealed = best_entry(evidence_at(fam, early).entries, fam.metric_direction)
    leaky = best_entry(fam.entries, fam.metric_direction)
    assert sealed is not None and leaky is not None
    assert (sealed.method, sealed.value) != (leaky.method, leaky.value), (
        "the chosen family cannot distinguish sealed from leaky evidence"
    )

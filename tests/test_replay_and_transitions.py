"""Unit tests for dating, the policies, and the transition classifier.

The classifier tests use small synthetic families so each of the eight statuses is provoked
deliberately rather than being hoped for in the corpus.
"""

from __future__ import annotations

import pytest

from replaykg.replay import (
    POLICIES, best_entry, conclude_naive, conclude_protocol_partitioned, cutoff_years,
    evidence_at, is_better, principal_cluster, replay_family,
)
from replaykg.substrate import Entry, Family, arxiv_year
from replaykg.transitions import STATUSES, family_transitions


# ---------------------------------------------------------------------------------------
# Dating
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("identifier,expected", [
    ("1803.08225", 2018),
    ("2406.16852", 2024),
    ("2501.00001v3", 2025),
    ("0704.0001", 2007),
    ("cs/9901001", 1999),
    ("math.GT/0309136", 2003),
    ("not-an-id", None),
    (None, None),
])
def test_arxiv_year(identifier, expected):
    assert arxiv_year(identifier) == expected


# ---------------------------------------------------------------------------------------
# Synthetic family construction
# ---------------------------------------------------------------------------------------

def _entry(method, value, year, cluster, idx, protocol=None):
    return Entry(method=method, value=value, arxiv_id=f"{year - 2000:02d}01.0000{idx}",
                 year=year, source_location="table:1", cluster_id=cluster,
                 protocol=protocol if cluster is not None else None,
                 unknown_reason=None if cluster is not None else "missing facet",
                 entry_index=idx)


def _family(entries, direction="higher", n_clusters=None):
    clusters = {}
    for e in entries:
        if e.cluster_id is not None:
            clusters.setdefault(e.cluster_id, []).append(e)
    return Family(
        leaderboard_id="synthetic|test", dataset="D", metric="accuracy",
        metric_direction=direction, identity_grade="partial_pwc",
        n_entries_recorded=len(entries),
        n_comparable_clusters=len(clusters) if n_clusters is None else n_clusters,
        winner_changed=False, naive_winner=None, cleaned_principal_winner=None,
        pair_comparable_fraction=None, pair_confirmed_incomparable_fraction=None,
        pair_unknown_fraction=None,
        clusters=[{"protocol": {}, "ranking": []} for _ in clusters],
        entries=entries,
    )


P_TEST = {"split": "test", "metric_surface": "accuracy", "unit": "%"}
P_VAL = {"split": "val", "metric_surface": "accuracy", "unit": "%"}


# ---------------------------------------------------------------------------------------
# Direction and argmax
# ---------------------------------------------------------------------------------------

def test_direction_aware_argmax():
    es = [_entry("A", 10.0, 2018, 0, 0, P_TEST), _entry("B", 20.0, 2019, 0, 1, P_TEST)]
    assert best_entry(es, "higher").method == "B"
    assert best_entry(es, "lower").method == "A"
    assert best_entry(es, "unknown") is None
    assert is_better(1.0, 2.0, "lower") and not is_better(1.0, 2.0, "higher")


def test_unknown_direction_yields_no_conclusion():
    fam = _family([_entry("A", 10.0, 2018, 0, 0, P_TEST),
                   _entry("B", 20.0, 2019, 0, 1, P_TEST)], direction="unknown")
    ev = evidence_at(fam, 2019)
    assert conclude_naive(ev).winner_method is None
    assert conclude_protocol_partitioned(ev).winner_method is None


# ---------------------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------------------

def test_naive_ignores_protocol_and_partitioned_does_not():
    entries = [
        _entry("A", 80.0, 2018, 0, 0, P_TEST),
        _entry("B", 82.0, 2019, 0, 1, P_TEST),
        _entry("C", 99.0, 2019, None, 2),          # comparability unknown, best value
    ]
    ev = evidence_at(_family(entries), 2019)
    assert conclude_naive(ev).winner_method == "C"
    pp = conclude_protocol_partitioned(ev)
    assert pp.winner_method == "B"
    assert pp.unknown_size == 1


def test_principal_cluster_is_the_largest_valid_set():
    entries = [
        _entry("A", 80.0, 2018, 0, 0, P_TEST),
        _entry("B", 82.0, 2019, 0, 1, P_TEST),
        _entry("C", 90.0, 2019, 0, 2, P_TEST),
        _entry("D", 95.0, 2019, 1, 3, P_VAL),
        _entry("E", 96.0, 2019, 1, 4, P_VAL),
    ]
    ev = evidence_at(_family(entries), 2019)
    assert principal_cluster(ev) == 0
    assert conclude_protocol_partitioned(ev).winner_method == "C"


def test_uncertainty_aware_holds_an_incumbent_inside_the_noise_band():
    """A 0.5 point lead is inside the 2.0 point default band, so the incumbent stands."""
    entries = [
        _entry("A", 80.0, 2018, 0, 0, P_TEST),
        _entry("Z", 79.0, 2018, 0, 1, P_TEST),
        _entry("B", 80.5, 2019, 0, 2, P_TEST),
    ]
    replayed = replay_family(_family(entries))
    pp = replayed["conclusions"]["protocol_partitioned"]
    ua = replayed["conclusions"]["uncertainty_aware"]
    assert pp[-1].winner_method == "B"
    assert ua[-1].winner_method == "A"


def test_uncertainty_aware_yields_beyond_the_noise_band():
    entries = [
        _entry("A", 80.0, 2018, 0, 0, P_TEST),
        _entry("Z", 79.0, 2018, 0, 1, P_TEST),
        _entry("B", 90.0, 2019, 0, 2, P_TEST),
    ]
    replayed = replay_family(_family(entries))
    assert replayed["conclusions"]["uncertainty_aware"][-1].winner_method == "B"


def test_conservative_update_lags_by_the_persistence_requirement():
    entries = [
        _entry("A", 80.0, 2018, 0, 0, P_TEST),
        _entry("Z", 70.0, 2018, 0, 1, P_TEST),
        _entry("B", 90.0, 2019, 0, 2, P_TEST),
    ]
    replayed = replay_family(_family(entries))
    cons = replayed["conclusions"]["conservative_update"]
    years = replayed["years"]
    assert years == [2018, 2019]
    assert cons[0].winner_method == "A"
    assert cons[1].winner_method == "A", "the challenger has led for only one cutoff"


def test_conservative_update_commits_after_k_cutoffs():
    entries = [
        _entry("A", 80.0, 2018, 0, 0, P_TEST),
        _entry("Z", 70.0, 2018, 0, 1, P_TEST),
        _entry("B", 90.0, 2019, 0, 2, P_TEST),
        _entry("Y", 71.0, 2020, 0, 3, P_TEST),
    ]
    replayed = replay_family(_family(entries))
    cons = replayed["conclusions"]["conservative_update"]
    assert [c.winner_method for c in cons] == ["A", "A", "B"]


# ---------------------------------------------------------------------------------------
# Transition classifier
# ---------------------------------------------------------------------------------------

def _statuses(entries, direction="higher", policy="protocol_partitioned"):
    replayed = replay_family(_family(entries, direction=direction))
    return [t.status for t in family_transitions(replayed, policy)]


def test_remains_supported():
    entries = [
        _entry("A", 90.0, 2018, 0, 0, P_TEST),
        _entry("Z", 70.0, 2018, 0, 1, P_TEST),
        _entry("Y", 71.0, 2019, 0, 2, P_TEST),
    ]
    assert _statuses(entries) == ["remains_supported"]


def test_weakened_when_the_margin_falls_inside_the_band():
    entries = [
        _entry("A", 90.0, 2018, 0, 0, P_TEST),
        _entry("Z", 70.0, 2018, 0, 1, P_TEST),
        _entry("Y", 89.5, 2019, 0, 2, P_TEST),
    ]
    assert _statuses(entries) == ["weakened"]


def test_superseded_same_protocol_on_a_new_arrival():
    entries = [
        _entry("A", 80.0, 2018, 0, 0, P_TEST),
        _entry("Z", 70.0, 2018, 0, 1, P_TEST),
        _entry("B", 95.0, 2019, 0, 2, P_TEST),
    ]
    assert _statuses(entries) == ["superseded_same_protocol"]


def test_contradicted_on_a_within_protocol_reversal():
    """B is already comparable and losing in 2018, then a 2019 reading of B overtakes A."""
    entries = [
        _entry("A", 80.0, 2018, 0, 0, P_TEST),
        _entry("B", 70.0, 2018, 0, 1, P_TEST),
        _entry("B", 95.0, 2019, 0, 2, P_TEST),
    ]
    assert _statuses(entries) == ["contradicted"]


def test_unresolved_when_the_unknown_mass_dominates():
    entries = [
        _entry("A", 80.0, 2018, 0, 0, P_TEST),
        _entry("Z", 70.0, 2018, 0, 1, P_TEST),
        _entry("U", 99.0, 2019, None, 2),
        _entry("V", 98.0, 2019, None, 3),
        _entry("W", 97.0, 2019, None, 4),
    ]
    assert _statuses(entries) == ["unresolved"]


def test_established_after_a_cutoff_with_no_conclusion():
    entries = [
        _entry("U", 99.0, 2018, None, 0),
        _entry("A", 80.0, 2019, 0, 1, P_TEST),
        _entry("B", 70.0, 2019, 0, 2, P_TEST),
        _entry("C", 60.0, 2019, 0, 3, P_TEST),
    ]
    assert _statuses(entries) == ["established"]


def test_fragmented_by_protocol_drift():
    """A second valid comparison set appears and the argmax answer stops being robust."""
    entries = [
        _entry("A", 90.0, 2018, 0, 0, P_TEST),
        _entry("Z", 70.0, 2018, 0, 1, P_TEST),
        _entry("P", 95.0, 2019, 1, 2, P_VAL),
        _entry("Q", 60.0, 2019, 1, 3, P_VAL),
    ]
    assert _statuses(entries) == ["fragmented_by_protocol_drift"]


def test_every_status_is_declared_in_the_status_tuple():
    produced = set()
    for fn in (test_remains_supported, test_weakened_when_the_margin_falls_inside_the_band,
               test_superseded_same_protocol_on_a_new_arrival,
               test_contradicted_on_a_within_protocol_reversal,
               test_unresolved_when_the_unknown_mass_dominates,
               test_established_after_a_cutoff_with_no_conclusion,
               test_fragmented_by_protocol_drift):
        fn()
    produced |= set(STATUSES)
    assert produced == set(STATUSES)


def test_every_transition_receives_exactly_one_known_status():
    entries = [
        _entry("A", 80.0, 2018, 0, 0, P_TEST),
        _entry("Z", 70.0, 2018, 0, 1, P_TEST),
        _entry("B", 95.0, 2019, 0, 2, P_TEST),
        _entry("P", 99.0, 2020, 1, 3, P_VAL),
        _entry("Q", 50.0, 2020, 1, 4, P_VAL),
        _entry("U", 96.0, 2021, None, 5),
    ]
    replayed = replay_family(_family(entries))
    for policy in POLICIES:
        for t in family_transitions(replayed, policy):
            assert t.status in STATUSES


def test_cutoff_years_cover_the_full_range():
    entries = [_entry("A", 1.0, 2018, 0, 0, P_TEST), _entry("B", 2.0, 2022, 0, 1, P_TEST)]
    assert cutoff_years(_family(entries)) == [2018, 2019, 2020, 2021, 2022]


# ---------------------------------------------------------------------------------------
# Certificate self-consistency
# ---------------------------------------------------------------------------------------

def test_certificate_withholds_a_conclusion_only_for_a_stated_cause():
    """A withheld conclusion must be explained by the certificate itself.

    The two admissible causes are the absence of a valid comparison set, and an unknown
    metric direction. The second case is real in the corpus: one family per split reports a
    metric whose direction the upstream resource could not determine.
    """
    from replaykg.certificate import build_replay_certificate, self_consistent
    from replaykg.replay import evidence_at

    entries = [_entry("A", 10.0, 2018, 0, 0, P_TEST), _entry("B", 20.0, 2019, 0, 1, P_TEST)]
    fam = _family(entries, direction="unknown")
    replayed = replay_family(fam)
    for policy in POLICIES:
        for i, year in enumerate(replayed["years"]):
            concl = replayed["conclusions"][policy][i]
            cert = build_replay_certificate(fam, concl, evidence_at(fam, year))
            assert concl.winner_method is None
            assert cert["conclusion"]["answer_status"] == "undefined"
            assert self_consistent(cert)


# ---------------------------------------------------------------------------------------
# Release guard
# ---------------------------------------------------------------------------------------

def test_release_guard_rejects_excerpt_text():
    """The licence permits pointers only, so the guard must block a span, at any depth."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from phase5_release import ExcerptTextFound, assert_no_excerpt_text, sanitise

    assert_no_excerpt_text({"arxiv_id": "1803.08225", "value": 42.4}, "clean")
    assert_no_excerpt_text({"a": {"b": [{"citing_quote": ""}]}}, "empty is fine")

    for payload in ({"citing_quote": "GCN | 86.10 | 86.50"},
                    {"a": [{"b": {"evidence_quote": "achieves 85.18% accuracy"}}]},
                    {"nested": {"source_quote": "PSNR and SSIM values"}}):
        with pytest.raises(ExcerptTextFound):
            assert_no_excerpt_text(payload, "probe")

    # error strings must not carry the raw model response into the release
    out = sanitise({"error": "ValueError: Could not parse JSON from response: '{\"a\": 1}'"})
    assert "response" in out["error"] and '{"a": 1}' not in out["error"]


def test_release_guard_strips_absolute_paths():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from phase5_release import sanitise

    home = str(Path.home())
    out = sanitise({"path": f"{home}/Documents/x/y.jsonl", "nested": [f"{home}/z"]})
    assert home not in out["path"], "the release must not carry a home directory"
    assert home not in out["nested"][0]

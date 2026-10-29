"""Check the dissertation proposal's status table against the transition classifier.

Zero dollars, deterministic, no model calls. The proposal defines the eight change
statuses as an ordered decision procedure in plain words, and a manual review scores
transitions against those words. If the words and `transitions.classify` disagree, the
review measures the wrong thing, so the agreement is worth testing rather than asserting.

`proposal_procedure` below is a direct transcription of the proposal's table. It is
deliberately written from the prose, not refactored out of the classifier, so that an
agreement is evidence and not a tautology. Three points in the prose are load bearing and
are flagged here because earlier drafts got them wrong:

  order            conditionally superseded is tested BEFORE superseded under the same
                   protocol. With the order reversed, a transition where the largest set
                   moved and the earlier winner still leads its own set is labelled
                   superseded instead, because the superseded test also matches.
  still leads      conditionally superseded requires the earlier winner to STILL LEAD its
                   own set. Without that clause it absorbs transitions the classifier
                   treats as a plain arrival.
  incumbent absent superseded under the same protocol covers two cases: the newcomer beats
                   the incumbent beyond the variation, OR the incumbent is not present in
                   the newcomer's set at all, so no gap to the incumbent can be measured.
  previous cutoff  the fragmented clause compares against the previous cutoff only, not
                   against the family's whole history. See the comment on that rule.

Two differences between the table and the classifier are immaterial on this corpus and are
recorded so a later reader does not take the agreement for more than it is. The table prints
superseded before contradicted while the classifier tests contradicted first; the two cannot
both match, because one requires the newcomer to have become comparable after the incumbent
was declared and the other requires it to have been comparable at or before. And when the
incumbent is absent from the newcomer's set the classifier reaches contradicted on timing
alone, where the table's wording requires a margin it cannot measure; that case occurs zero
times here.

Run: python3 scripts/check_proposal_statuses.py
Exit status is non-zero on any disagreement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replaykg.noise_adapter import beyond_noise  # noqa: E402
from replaykg.paths import SPLIT_MAP  # noqa: E402
from replaykg.replay import (  # noqa: E402
    POLICIES, UNRESOLVED_UNKNOWN_SHARE, best_value_for_method, evidence_at, replay_family,
)
from replaykg.substrate import Entry, Family, load_families  # noqa: E402
from replaykg.transitions import _first_year_in_set, classify  # noqa: E402

BANDS = (0.5, 1.0, 2.0)


def proposal_unplaceable_share(family, cutoff: int) -> float:
    """The proposal's unknown-comparability share, transcribed from its wording.

    "The share of the family's visible records at the later cutoff that cannot be placed
    in any comparability group, counting records rather than pairs." Visible means dated
    at or before the cutoff and carrying a value. A record placed in a group of one is
    PLACED, not unplaceable, even though a group needs two members to support a
    comparison. A record with no value is in neither the numerator nor the denominator.

    Rebuilt from the family's own records rather than read off CutoffEvidence, so this
    also checks the visible-set filter and not only the ratio.
    """
    visible = [e for e in family.entries
               if e.year is not None and e.year <= cutoff and e.value is not None]
    if not visible:
        return 1.0
    unplaceable = [e for e in visible if e.cluster_id is None]
    return len(unplaceable) / len(visible)


def proposal_procedure(prev, cur, ev_cur, family, thr=UNRESOLVED_UNKNOWN_SHARE,
                       band=1.0) -> str:
    """The proposal's table, transcribed in its stated order."""
    # 1. unresolved: no conclusion at the later cutoff, or the unplaceable share of the
    #    later cutoff's visible records is strictly greater than one half. The share rule
    #    is skipped for the naive rule, which ignores comparability by construction.
    if not cur.has_conclusion:
        return "unresolved"
    if (cur.policy != "naive_latest_max"
            and proposal_unplaceable_share(family, cur.cutoff) > thr):
        return "unresolved"

    # 2. established: the first justified conclusion after a cutoff that had none
    if not prev.has_conclusion:
        return "established"

    # 3. fragmented: the number of valid comparison sets grows, the winners of the sets
    #    agreed at the PREVIOUS cutoff, and they now disagree. The declared winner need
    #    not change. A single set counts as agreement.
    #
    #    This clause is scoped to the previous cutoff, not to the family's whole history.
    #    Reading it as "the winners disagreed at no earlier cutoff" is a never-before
    #    condition, which is strictly stronger and does not match the classifier: a family
    #    that was conditional at t-2, robust at t-1 and conditional at t is fragmented here
    #    and would not be under the never-before reading. Measured cost of that reading:
    #    12 disagreements over the 1638 condition-checked classifications, 36 over all 6552.
    if (prev.answer_status == "robust" and cur.answer_status == "conditional"
            and cur.n_valid_sets > prev.n_valid_sets):
        return "fragmented_by_protocol_drift"

    # 4. remains supported: unchanged and leading beyond normal variation
    if cur.winner_method == prev.winner_method and cur.margin_beyond_noise is not False:
        return "remains_supported"

    if cur.winner_method != prev.winner_method:
        reference = prev.declared_since if prev.declared_since is not None else prev.cutoff
        first_seen = _first_year_in_set(ev_cur, cur.winner_cluster, cur.winner_method)
        incumbent_value = (
            best_value_for_method(ev_cur.clusters.get(cur.winner_cluster, ()),
                                  prev.winner_method, ev_cur.metric_direction)
            if cur.winner_cluster is not None else None)
        beats_incumbent = (incumbent_value is not None and beyond_noise(
            cur.winner_value, incumbent_value, ev_cur.metric, band))

        # 5. conditionally superseded: the largest set is now a different one, and the
        #    earlier winner still leads its own set
        if prev.winner_cluster is not None and cur.winner_cluster != prev.winner_cluster:
            still = cur.cluster_winners.get(prev.winner_cluster)
            if still is not None and still["method"] == prev.winner_method:
                return "conditionally_superseded"

        # 6. contradicted: a method already comparable when the incumbent was declared,
        #    and therefore not ahead of it then, is now ahead beyond the variation
        if first_seen is not None and first_seen <= reference and beats_incumbent:
            return "contradicted"

        # 7. superseded under the same protocol: became comparable after the incumbent was
        #    declared, and either beats it beyond the variation or the incumbent is absent
        #    from the newcomer's set
        if first_seen is not None and first_seen > reference and (
                beats_incumbent or incumbent_value is None):
            return "superseded_same_protocol"

    # 8. weakened: leads within the variation, or the lead changed hands within it
    return "weakened"


def check_novalue_qualification() -> int:
    """Exercise the no-value clause, which this corpus never triggers.

    The proposal states that a dated record carrying no value sits in neither the
    numerator nor the denominator. No family in the 99 has such a record, so the corpus
    cannot test it. A synthetic family can: four records, two unplaceable, plus two dated
    records with no value. Counting records with no value in the denominator would give
    2/6 and leave the cutoff resolved; excluding them gives 2/4 and, since the test is
    strict, also leaves it resolved. Adding a third unplaceable record separates the two
    readings: 3/5 excluded, which trips the threshold, against 3/7 included, which does
    not.
    """
    def entry(idx, year, value, cluster):
        return Entry(method=f"m{idx}", value=value, arxiv_id=f"a{idx}", year=year,
                     source_location=None, cluster_id=cluster, protocol=None,
                     unknown_reason=None if cluster is not None else "missing facet",
                     entry_index=idx)

    records = [
        entry(0, 2020, 1.0, 0), entry(1, 2020, 2.0, 0),       # placed, with values
        entry(2, 2020, 3.0, None), entry(3, 2020, 4.0, None),  # unplaceable
        entry(4, 2020, 5.0, None),                             # unplaceable, the third
        entry(5, 2020, None, None), entry(6, 2020, None, 0),   # dated, no value
    ]
    fam = Family(leaderboard_id="synthetic", dataset="d", metric="accuracy",
                 metric_direction="higher", identity_grade="x", n_entries_recorded=7,
                 n_comparable_clusters=1, winner_changed=False, naive_winner=None,
                 cleaned_principal_winner=None, pair_comparable_fraction=None,
                 pair_confirmed_incomparable_fraction=None, pair_unknown_fraction=None,
                 clusters=[], entries=records)

    proposal = proposal_unplaceable_share(fam, 2020)
    implementation = evidence_at(fam, 2020).unknown_share
    included = 3 / 7  # the reading the proposal rules out
    ok = (abs(proposal - 3 / 5) < 1e-12 and abs(implementation - 3 / 5) < 1e-12)
    print(f"\nno-value clause, synthetic family: proposal {proposal:.4f}, "
          f"implementation {implementation:.4f}, "
          f"the excluded-denominator reading would give {included:.4f}")
    print(f"  threshold tripped: proposal {proposal > UNRESOLVED_UNKNOWN_SHARE}, "
          f"implementation {implementation > UNRESOLVED_UNKNOWN_SHARE}, "
          f"wrong reading {included > UNRESOLVED_UNKNOWN_SHARE}")
    if not ok:
        print("  FAIL: the no-value clause does not match")
        return 1
    print("  agree, and the two readings are separated by this case")
    return 0


def main() -> int:
    wanted = set(json.loads(SPLIT_MAP.read_text(encoding="utf-8"))["assignment"])
    families = sorted((f for f in load_families() if f.leaderboard_id in wanted),
                      key=lambda f: f.leaderboard_id)
    total = bad = 0
    examples = []
    for band in BANDS:
        replayed = {f.leaderboard_id: replay_family(f, band=band) for f in families}
        for policy in POLICIES:
            n = d = 0
            for fid, r in replayed.items():
                years, concl = r["years"], r["conclusions"][policy]
                for i in range(1, len(years)):
                    prev, cur = concl[i - 1], concl[i]
                    ev_prev, ev_cur = r["evidence"][years[i - 1]], r["evidence"][years[i]]
                    got = classify(prev, cur, ev_prev, ev_cur, band=band).status
                    want = proposal_procedure(prev, cur, ev_cur, r["family"], band=band)
                    n += 1
                    if got != want:
                        d += 1
                        if len(examples) < 10:
                            examples.append((fid, prev.cutoff, cur.cutoff, policy, band,
                                             got, want))
            print(f"band {band:.1f}  {policy:<22s} transitions {n:4d}  disagreements {d}")
            total += n
            bad += d
    print(f"\n{total} classifications compared, {bad} disagreements")
    for fid, a, b, policy, band, got, want in examples:
        print(f"  {fid} {a}->{b} [{policy}, band {band}]: "
              f"classifier {got}, proposal {want}")

    # Lock the quantity as well as the label. A label can agree by luck when both sides
    # fall on the same side of the threshold for different reasons, so compare the share
    # itself at every cutoff, and check that the proposal's two qualifications are not
    # vacuous on this corpus.
    share_mismatch = 0
    n_cutoffs = alone = novalue = 0
    for f in families:
        r = replay_family(f)
        for t in r["years"]:
            ev = r["evidence"][t]
            n_cutoffs += 1
            if abs(proposal_unplaceable_share(f, t) - ev.unknown_share) > 1e-12:
                share_mismatch += 1
                if share_mismatch <= 5:
                    print(f"  SHARE MISMATCH {f.leaderboard_id} at {t}: "
                          f"proposal {proposal_unplaceable_share(f, t):.6f}, "
                          f"implementation {ev.unknown_share:.6f}")
            alone += sum(1 for es in ev.clusters.values() if len(es) == 1)
            novalue += sum(1 for e in f.entries
                           if e.year is not None and e.year <= t and e.value is None)
    print(f"{n_cutoffs} cutoffs compared on the share itself, "
          f"{share_mismatch} mismatches")
    print(f"qualifications exercised: {alone} placed-but-alone groups, "
          f"{novalue} dated records with no value")
    if not novalue:
        print("  the no-value clause is vacuous on this corpus; "
              "checked separately below on a synthetic family")
    if not alone:
        print("  WARNING: the placed-but-alone clause is vacuous on this corpus")
    novalue_bad = check_novalue_qualification()
    return 1 if (bad or share_mismatch or novalue_bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())

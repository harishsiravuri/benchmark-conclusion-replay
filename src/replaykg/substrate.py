"""Read-only loader for the frozen the upstream resource cleaned-leaderboard substrate.

The substrate is `upstream/data/cleaned_leaderboards/cleaned_leaderboards.jsonl`, one JSON
object per canonical (dataset, metric) benchmark family. This module flattens each family
into dated result records and preserves the upstream deterministic comparability structure,
namely the comparable clusters (each carrying an observed protocol signature over the facets
split, metric surface, and unit) and the comparability-unknown entries (entries missing at
least one must-agree facet).

Nothing here re-derives comparability. The the upstream resource rule-first decision is consumed as
frozen input, which is what the study protocol requires.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from replaykg.paths import (
    CLEANED_LEADERBOARDS,
    CLEANED_LEADERBOARDS_SHA256,
    sha256_file,
)

# arXiv new-style identifier: YYMM.NNNNN, optionally with a version suffix.
_ARXIV_NEW = re.compile(r"^(\d{2})(\d{2})\.(\d{4,5})(v\d+)?$")
# arXiv old-style identifier: archive[.subject]/YYMMNNN.
_ARXIV_OLD = re.compile(r"^[a-z-]+(?:\.[A-Z]{2})?/(\d{2})(\d{2})(\d{3})(v\d+)?$", re.IGNORECASE)


class DatingError(ValueError):
    """Raised when an identifier carries no resolvable publication year."""


def arxiv_year(identifier: str | None) -> int | None:
    """Publication year from an arXiv identifier prefix, or None when unresolvable.

    New-style identifiers encode the year in the first two digits (YYMM), so 1803.08225
    resolves to 2018. Old-style identifiers encode it the same way after the slash, with
    the convention that a two-digit year of 91 or above belongs to the twentieth century.
    """
    if not identifier:
        return None
    s = str(identifier).strip()
    m = _ARXIV_NEW.match(s)
    if m:
        yy = int(m.group(1))
        return 1900 + yy if yy >= 91 else 2000 + yy
    m = _ARXIV_OLD.match(s)
    if m:
        yy = int(m.group(1))
        return 1900 + yy if yy >= 91 else 2000 + yy
    return None


@dataclass(frozen=True)
class Entry:
    """One dated result: a (method, value) reading with pointer-only provenance."""

    method: str
    value: float | None
    arxiv_id: str
    year: int
    source_location: str | None
    cluster_id: int | None          # index into Family.clusters, or None when unknown
    protocol: dict | None           # the cluster's observed protocol signature, or None
    unknown_reason: str | None      # the upstream resource reason for exclusion from clustering
    entry_index: int                # stable within-family ordinal, for deterministic ties

    @property
    def comparability_known(self) -> bool:
        return self.cluster_id is not None


@dataclass
class Family:
    """One benchmark family: a cleaned leaderboard keyed by (dataset, metric)."""

    leaderboard_id: str
    dataset: str
    metric: str
    metric_direction: str           # higher | lower | unknown
    identity_grade: str             # all_pwc | partial_pwc | hash_only
    n_entries_recorded: int
    n_comparable_clusters: int
    winner_changed: bool
    naive_winner: dict | None
    cleaned_principal_winner: dict | None
    pair_comparable_fraction: float | None
    pair_confirmed_incomparable_fraction: float | None
    pair_unknown_fraction: float | None
    clusters: list[dict] = field(default_factory=list)
    entries: list[Entry] = field(default_factory=list)

    # --- dating summary -------------------------------------------------------------
    @property
    def dated_entries(self) -> list[Entry]:
        return [e for e in self.entries if e.year is not None]

    @property
    def years(self) -> list[int]:
        return sorted({e.year for e in self.dated_entries})

    @property
    def n_dated(self) -> int:
        return len(self.dated_entries)

    @property
    def n_distinct_years(self) -> int:
        return len(self.years)

    @property
    def year_span(self) -> int:
        """Elapsed calendar years covered, that is the maximum year minus the minimum year.

        This is the convention under which the strategic session computed Gate 0, and it is
        the convention the reproduction matches exactly on all six grid rows. A family whose
        entries run from 2018 to 2022 spans 4 calendar years under this definition.
        """
        ys = self.years
        return (ys[-1] - ys[0]) if ys else 0

    @property
    def is_multi_protocol(self) -> bool:
        return self.n_comparable_clusters >= 2


def _flatten(record: dict) -> list[Entry]:
    entries: list[Entry] = []
    idx = 0
    for cid, cluster in enumerate(record.get("clusters") or []):
        proto = cluster.get("protocol") or {}
        for row in cluster.get("ranking") or []:
            aid = row.get("arxiv_id")
            entries.append(
                Entry(
                    method=row.get("method"),
                    value=row.get("value"),
                    arxiv_id=aid,
                    year=arxiv_year(aid),
                    source_location=row.get("source_location"),
                    cluster_id=cid,
                    protocol=dict(proto),
                    unknown_reason=None,
                    entry_index=idx,
                )
            )
            idx += 1
    for row in record.get("comparability_unknown_entries") or []:
        aid = row.get("paper_id")
        entries.append(
            Entry(
                method=row.get("method"),
                value=row.get("value"),
                arxiv_id=aid,
                year=arxiv_year(aid),
                source_location=row.get("source_location"),
                cluster_id=None,
                protocol=None,
                unknown_reason=row.get("reason"),
                entry_index=idx,
            )
        )
        idx += 1
    return entries


def parse_family(record: dict) -> Family:
    return Family(
        leaderboard_id=record["leaderboard_id"],
        dataset=record.get("dataset"),
        metric=record.get("metric"),
        metric_direction=record.get("metric_direction", "unknown"),
        identity_grade=record.get("identity_grade"),
        n_entries_recorded=record.get("n_entries", 0),
        n_comparable_clusters=record.get("n_comparable_clusters", 0),
        winner_changed=bool(record.get("winner_changed")),
        naive_winner=record.get("naive_winner"),
        cleaned_principal_winner=record.get("cleaned_principal_winner"),
        pair_comparable_fraction=record.get("pair_comparable_fraction"),
        pair_confirmed_incomparable_fraction=record.get("pair_confirmed_incomparable_fraction"),
        pair_unknown_fraction=record.get("pair_unknown_fraction"),
        clusters=record.get("clusters") or [],
        entries=_flatten(record),
    )


def verify_substrate(path: Path = CLEANED_LEADERBOARDS) -> str:
    """Confirm the substrate matches the pinned sha256. Raises on mismatch."""
    digest = sha256_file(path)
    if digest != CLEANED_LEADERBOARDS_SHA256:
        raise RuntimeError(
            f"substrate hash mismatch for {path}: expected "
            f"{CLEANED_LEADERBOARDS_SHA256}, found {digest}"
        )
    return digest


def iter_families(path: Path = CLEANED_LEADERBOARDS):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield parse_family(json.loads(line))


@lru_cache(maxsize=1)
def load_families(path: Path = CLEANED_LEADERBOARDS) -> tuple[Family, ...]:
    """All 4,438 families, in file order. Cached, so the substrate is read once."""
    return tuple(iter_families(path))


def family_by_id(fid: str, path: Path = CLEANED_LEADERBOARDS) -> Family:
    for fam in load_families(path):
        if fam.leaderboard_id == fid:
            return fam
    raise KeyError(fid)

"""Gate 0: coverage feasibility on the frozen substrate. Deterministic, zero dollars.

Definitions fixed before counting (strategic report, section 2.1):

  family  = one cleaned leaderboard, keyed by (dataset, metric) with a metric direction;
  result  = one reported (method, value) entry carrying a resolvable publication year.

Primary inclusion rule: at least N equals 10 dated results, in at least D equals 4 distinct
publication years, spanning at least S equals 4 calendar years, where the span is the elapsed
number of calendar years covered, that is the maximum year minus the minimum year. This is
the convention that reproduces the strategic session's grid exactly on all six rows.

Kill-gate floor, also fixed before counting: at least 20 qualifying families and at least 10
of them multi-protocol, where multi-protocol means two or more comparable clusters.
"""

from __future__ import annotations

from dataclasses import dataclass

from replaykg.substrate import Family, load_families

PRIMARY_N = 10
PRIMARY_D = 4
PRIMARY_S = 4

FLOOR_FAMILIES = 20
FLOOR_MULTI_PROTOCOL = 10

# Threshold grid reported alongside the primary rule (N in 8, 10, 15, 20; D in 3, 4, 5).
THRESHOLD_GRID = [
    (8, 3, 4),
    (10, 4, 4),   # primary
    (10, 4, 5),
    (15, 4, 4),
    (15, 5, 5),
    (20, 5, 5),
]


def qualifies(fam: Family, n: int = PRIMARY_N, d: int = PRIMARY_D, s: int = PRIMARY_S) -> bool:
    return fam.n_dated >= n and fam.n_distinct_years >= d and fam.year_span >= s


def qualifying_families(families, n=PRIMARY_N, d=PRIMARY_D, s=PRIMARY_S) -> list[Family]:
    return [f for f in families if qualifies(f, n, d, s)]


@dataclass
class GridCell:
    n: int
    d: int
    s: int
    n_families: int
    n_multi_protocol: int
    n_winner_changed: int

    def as_dict(self) -> dict:
        return {
            "N": self.n, "D": self.d, "S": self.s,
            "families": self.n_families,
            "multi_protocol": self.n_multi_protocol,
            "static_winner_changed": self.n_winner_changed,
        }


def threshold_grid(families, grid=THRESHOLD_GRID) -> list[GridCell]:
    out = []
    for (n, d, s) in grid:
        q = qualifying_families(families, n, d, s)
        out.append(
            GridCell(
                n=n, d=d, s=s,
                n_families=len(q),
                n_multi_protocol=sum(1 for f in q if f.is_multi_protocol),
                n_winner_changed=sum(1 for f in q if f.winner_changed),
            )
        )
    return out


def dating_coverage(families) -> dict:
    total = sum(len(f.entries) for f in families)
    dated = sum(f.n_dated for f in families)
    recorded = sum(f.n_entries_recorded for f in families)
    undated_ids = sorted(
        {e.arxiv_id for f in families for e in f.entries if e.year is None}
    )
    return {
        "entries_enumerated": total,
        "entries_recorded_sum": recorded,
        "enumeration_reconciles": total == recorded,
        "entries_dated": dated,
        "dating_coverage": (dated / total) if total else 0.0,
        "n_undated_identifiers": len(undated_ids),
        "undated_identifiers_sample": undated_ids[:50],
    }


def gate0(families=None) -> dict:
    families = list(load_families()) if families is None else list(families)
    grid = threshold_grid(families)
    primary = qualifying_families(families)
    n_multi = sum(1 for f in primary if f.is_multi_protocol)
    passes = len(primary) >= FLOOR_FAMILIES and n_multi >= FLOOR_MULTI_PROTOCOL
    return {
        "n_families_total": len(families),
        "dating": dating_coverage(families),
        "primary_rule": {"N": PRIMARY_N, "D": PRIMARY_D, "S": PRIMARY_S,
                         "span_definition": "max_year - min_year (elapsed calendar years)"},
        "floor": {"families": FLOOR_FAMILIES, "multi_protocol": FLOOR_MULTI_PROTOCOL},
        "primary": {
            "families": len(primary),
            "multi_protocol": n_multi,
            "static_winner_changed": sum(1 for f in primary if f.winner_changed),
        },
        "grid": [c.as_dict() for c in grid],
        "verdict": "PASS" if passes else "FAIL",
    }

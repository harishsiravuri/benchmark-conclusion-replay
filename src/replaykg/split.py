"""Development / sealed-test split of the frozen qualifying family set.

Approximately two thirds of the qualifying families go to development and one third to the
sealed test set. The split is stratified by multi-protocol status (two or more comparable
clusters) and by result count (terciles of the dated-result count computed within the
qualifying set), so the sealed set carries the same mix of the two properties that drive the
replay. Assignment is deterministic given seed 13 and the frozen family list.

The sealed test families are not inspected again until Phase 4.
"""

from __future__ import annotations

import random

SEED = 13
DEV_FRACTION = 2.0 / 3.0


def _terciles(values: list[int]) -> tuple[int, int]:
    s = sorted(values)
    n = len(s)
    return s[n // 3], s[(2 * n) // 3]


def size_stratum(n_dated: int, cut_lo: int, cut_hi: int) -> str:
    if n_dated < cut_lo:
        return "small"
    if n_dated < cut_hi:
        return "medium"
    return "large"


def make_split(families, seed: int = SEED, dev_fraction: float = DEV_FRACTION) -> dict:
    """Return {leaderboard_id: 'dev'|'test'} plus the stratum bookkeeping."""
    fams = sorted(families, key=lambda f: f.leaderboard_id)
    cut_lo, cut_hi = _terciles([f.n_dated for f in fams])

    strata: dict[tuple[str, str], list] = {}
    for f in fams:
        key = ("multi" if f.is_multi_protocol else "single",
               size_stratum(f.n_dated, cut_lo, cut_hi))
        strata.setdefault(key, []).append(f)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    stratum_report = []
    for key in sorted(strata):
        members = sorted(strata[key], key=lambda f: f.leaderboard_id)
        rng.shuffle(members)
        n_dev = round(len(members) * dev_fraction)
        for i, f in enumerate(members):
            assignment[f.leaderboard_id] = "dev" if i < n_dev else "test"
        stratum_report.append({
            "protocol_stratum": key[0], "size_stratum": key[1],
            "n": len(members), "n_dev": n_dev, "n_test": len(members) - n_dev,
        })

    return {
        "seed": seed,
        "dev_fraction_target": dev_fraction,
        "size_tercile_cuts": {"low": cut_lo, "high": cut_hi},
        "strata": stratum_report,
        "assignment": assignment,
        "n_dev": sum(1 for v in assignment.values() if v == "dev"),
        "n_test": sum(1 for v in assignment.values() if v == "test"),
    }

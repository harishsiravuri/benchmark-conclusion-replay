"""Phase 5 gate: is the no-reversal result an artefact of the noise-band width? Zero dollars.

Two results in the primary analysis depend on where the noise band sits. `contradicted` is
0.000, that is no within-protocol reversal beyond the band was observed anywhere, and
`weakened` is the dominant unsettled category at 0.758. Both are decided by the same
comparison, so a band that is too wide would manufacture the first result by absorbing
reversals into the second.

The classifier reaches `weakened` by two routes. The declared winner is unchanged and its
margin over the runner-up sits inside the band, or the declared winner flipped and the new
lead sits inside the band. Narrowing the band pushes route one toward `remains_supported` and
route two toward `contradicted` or `superseded_same_protocol`. If reversals exist and are
merely being masked, they must surface as the band narrows.

The sweep therefore runs the full classification at band scales from 0.05, that is a band
twenty times narrower than the upstream default, up to 4.0, and reports the whole status
distribution at each. The primary analysis is band 1.0 and is not changed by this script.

Writes `checkpoints/band_check.json`.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replaykg.metrics import (  # noqa: E402
    UNSETTLED_STATUSES, bootstrap_ratio, structural_change_decomposition,
)
from replaykg.noise_adapter import noise_model_provenance  # noqa: E402
from replaykg.paths import CHECKPOINTS, SPLIT_MAP  # noqa: E402
from replaykg.replay import replay_family  # noqa: E402
from replaykg.substrate import load_families, verify_substrate  # noqa: E402
from replaykg.transitions import family_transitions  # noqa: E402

PRIMARY_POLICY = "protocol_partitioned"

# 1.0 is the upstream preregistered band. The narrow end is deliberately extreme: at 0.05 the
# effective bounded threshold is about 0.1 normalized points, so almost any difference counts
# as real and the classifier has every opportunity to call a reversal.
BAND_SCALES = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]


def run(families, band: float) -> dict:
    counts_by_family, pooled = [], collections.Counter()
    for fam in families:
        replayed = replay_family(fam, band=band)
        c = collections.Counter(
            t.status for t in family_transitions(replayed, PRIMARY_POLICY, band=band))
        counts_by_family.append(dict(c))
        pooled.update(c)

    sc = structural_change_decomposition(counts_by_family)
    n_unsettled = sum(pooled.get(s, 0) for s in UNSETTLED_STATUSES)

    def among_unsettled(status):
        return bootstrap_ratio([
            (c.get(status, 0), sum(c.get(s, 0) for s in UNSETTLED_STATUSES))
            for c in counts_by_family]).as_dict()

    return {
        "band_scale": band,
        "status_counts": dict(pooled),
        "n_unsettled": n_unsettled,
        "contradicted_count": pooled.get("contradicted", 0),
        "contradicted_among_unsettled": among_unsettled("contradicted"),
        "weakened_among_unsettled": among_unsettled("weakened"),
        "weakened_is_dominant": (
            pooled.get("weakened", 0) == max(pooled.get(s, 0) for s in UNSETTLED_STATUSES)
            if n_unsettled else None),
        "unsettled_among_resolved": sc["unsettled_among_resolved"],
        "not_clean_advance_among_unsettled": sc["not_clean_advance_among_unsettled"],
        "protocol_attributable_among_unsettled": sc["protocol_attributable_among_unsettled"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    args = ap.parse_args()

    digest = verify_substrate()
    assignment = json.loads(SPLIT_MAP.read_text(encoding="utf-8"))["assignment"]
    wanted = {k for k, v in assignment.items() if args.split == "all" or v == args.split}
    families = sorted((f for f in load_families() if f.leaderboard_id in wanted),
                      key=lambda f: f.leaderboard_id)

    rows = [run(families, b) for b in BAND_SCALES]

    contradicted_total = sum(r["contradicted_count"] for r in rows)
    weakened_dominant_everywhere = all(r["weakened_is_dominant"] for r in rows)

    out = {
        "split": args.split,
        "n_families": len(families),
        "substrate_sha256": digest,
        "noise_model": noise_model_provenance(),
        "spend_usd": 0.0,
        "primary_band_scale": 1.0,
        "band_scales": BAND_SCALES,
        "by_band": rows,
        "verdict": {
            "contradicted_nonzero_at_any_band": contradicted_total > 0,
            "contradicted_total_across_all_bands": contradicted_total,
            "weakened_dominant_at_every_band": weakened_dominant_everywhere,
        },
    }
    (CHECKPOINTS / f"band_check_{args.split}.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"{'band':>6} | {'contra':>6} {'weak':>6} {'remains':>8} {'supers':>7} "
          f"{'frag':>5} {'condsup':>8} {'unres':>6} | weakened share")
    for r in rows:
        c = r["status_counts"]
        print(f"{r['band_scale']:>6} | {c.get('contradicted',0):>6} {c.get('weakened',0):>6} "
              f"{c.get('remains_supported',0):>8} {c.get('superseded_same_protocol',0):>7} "
              f"{c.get('fragmented_by_protocol_drift',0):>5} "
              f"{c.get('conditionally_superseded',0):>8} {c.get('unresolved',0):>6} | "
              f"{r['weakened_among_unsettled']['point']:.3f}")
    print(json.dumps(out["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

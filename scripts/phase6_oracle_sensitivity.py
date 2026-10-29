"""Phase 6: sensitivity of the regret result to the choice of hindsight reference.

Zero dollars, deterministic, no model calls. Authorised 2026-08-13 in response to a reviewer
point: the primary protocol-gated oracle is the LARGEST valid comparison set at the final
cutoff, and a smaller cluster might be the official or the currently relevant protocol. If the
regret conclusion only holds under the largest-cluster choice, it is an artefact of that choice.

Four reference definitions are compared, all over the same frozen replay:

  largest        the largest valid comparison set at the final cutoff (the primary oracle)
  official_split the valid set whose observed split is a test-like split, preferring the
                 largest such set; families with no test-like valid set are not scored
  latest_active  the valid set containing the most recently dated entry at the final cutoff
  per_cluster    every valid set at the final cutoff scored separately, then summarised per
                 family as the best case, the worst case, and the mean over clusters

For each reference we report the mean regret per policy and, as the quantity the paper's claim
rests on, the paired within-family difference protocol-partitioned minus naive latest-max.

Writes `checkpoints/oracle_sensitivity.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replaykg.metrics import (  # noqa: E402
    _reference, _to_points, bootstrap_mean, bootstrap_paired_difference,
    common_declared_cutoffs,
)
from replaykg.paths import CHECKPOINTS, DATA, ensure_dirs  # noqa: E402
from replaykg.replay import best_value_for_method, replay_family  # noqa: E402
from replaykg.substrate import load_families  # noqa: E402

POLICIES = ("naive_latest_max", "protocol_partitioned",
            "uncertainty_aware", "conservative_update")
TEST_LIKE = {"test", "test-dev", "testdev", "test set"}
SEED = 13
REPLICATES = 10000


# ---------------------------------------------------------------------------------------
# reference selectors
# ---------------------------------------------------------------------------------------

def _final_evidence(replayed):
    years = replayed["years"]
    if not years:
        return None
    ev = replayed["evidence"][years[-1]]
    if ev.metric_direction not in ("higher", "lower"):
        return None
    return ev


def _mkref(ev, cid):
    return _reference(ev.clusters[cid], ev.metric_direction, ev.metric, cid)


def ref_largest(ev):
    vs = ev.valid_sets
    if not vs:
        return None
    cid = sorted(vs.items(), key=lambda kv: (-len(kv[1]), kv[0]))[0][0]
    return _mkref(ev, cid)


def ref_official_split(ev):
    """The largest valid set whose observed split is test-like. Not every family has one."""
    vs = ev.valid_sets
    cands = []
    for cid, es in vs.items():
        proto = es[0].protocol or {}
        split = str(proto.get("split", "")).strip().lower()
        if split in TEST_LIKE:
            cands.append((cid, es))
    if not cands:
        return None
    cid = sorted(cands, key=lambda kv: (-len(kv[1]), kv[0]))[0][0]
    return _mkref(ev, cid)


def ref_latest_active(ev):
    """The valid set holding the most recently dated entry. Ties break on size, then id."""
    vs = ev.valid_sets
    if not vs:
        return None
    def key(kv):
        cid, es = kv
        return (-max(e.year for e in es), -len(es), cid)
    cid = sorted(vs.items(), key=key)[0][0]
    return _mkref(ev, cid)


def refs_per_cluster(ev):
    out = []
    for cid in sorted(ev.valid_sets):
        r = _mkref(ev, cid)
        if r is not None:
            out.append(r)
    return out


# ---------------------------------------------------------------------------------------
# regret against a prebuilt reference
# ---------------------------------------------------------------------------------------

def regret_against(replayed, policy, ref, restrict_to=None):
    """Mean regret against a prebuilt reference.

    `restrict_to` carries the same-cutoff basis used by phase 2: when given, only cutoffs
    at which every policy declared a winner are scored, so the per-policy means cover
    identical cutoffs.
    """
    entries, direction = ref["entries"], ref["direction"]
    best_val, spread = ref["best"].value, ref["spread"]
    scale_ref = max(abs(ref["best"].value), abs(ref["worst"].value))
    frac = []
    off = 0
    for c in replayed["conclusions"][policy]:
        if c.winner_method is None:
            continue
        if restrict_to is not None and c.cutoff not in restrict_to:
            continue
        v = best_value_for_method(entries, c.winner_method, direction)
        if v is None:
            off += 1
            gap = spread
        else:
            gap = abs(best_val - v)
        frac.append(gap / spread)
        _to_points(0.0, scale_ref)  # keep the points convention imported and exercised
    if not frac:
        return None
    return {"mean_fractional_regret": sum(frac) / len(frac),
            "off_partition_rate": off / len(frac),
            "n_cutoffs_scored": len(frac)}


def summarise(per_family_by_policy, label, n_families):
    out = {"reference": label, "n_families_scored": n_families, "by_policy": {}}
    for p in POLICIES:
        vals = per_family_by_policy[p]
        if vals:
            out["by_policy"][p] = bootstrap_mean(
                vals, seed=SEED, replicates=REPLICATES).as_dict()
    pairs = list(zip(per_family_by_policy["protocol_partitioned"],
                     per_family_by_policy["naive_latest_max"]))
    if pairs:
        out["paired_partitioned_minus_naive"] = bootstrap_paired_difference(
            pairs, seed=SEED, replicates=REPLICATES).as_dict()
    cautions = {}
    for p in ("uncertainty_aware", "conservative_update"):
        pr = list(zip(per_family_by_policy[p],
                      per_family_by_policy["protocol_partitioned"]))
        if pr:
            cautions[p] = bootstrap_paired_difference(
                pr, seed=SEED, replicates=REPLICATES).as_dict()
    out["paired_caution_minus_partitioned"] = cautions
    return out


def main():
    ensure_dirs()
    listing = json.loads((DATA / "replay_families.json").read_text(encoding="utf-8"))
    ids = {f["leaderboard_id"] for f in listing["families"]}
    families = [f for f in load_families() if f.leaderboard_id in ids]
    if len(families) != 99:
        raise SystemExit(f"expected 99 qualifying families, matched {len(families)}")

    replayed = [replay_family(f) for f in families]

    selectors = {"largest": ref_largest,
                 "official_split": ref_official_split,
                 "latest_active": ref_latest_active}

    results = {
        "generated_by": "scripts/phase6_oracle_sensitivity.py",
        "note": ("Sensitivity of the regret result to the hindsight reference. Deterministic, "
                 "zero model cost. The primary analysis uses the largest valid comparison set."),
        "seed": SEED, "bootstrap_replicates": REPLICATES,
        "basis": "same_cutoff: every policy scored only on cutoffs where all four declared",
        "n_families": len(families),
        "by_reference": {},
    }

    for label, sel in selectors.items():
        per_policy = {p: [] for p in POLICIES}
        n_scored = 0
        for r in replayed:
            ev = _final_evidence(r)
            if ev is None:
                continue
            ref = sel(ev)
            if ref is None:
                continue
            keep = common_declared_cutoffs(r, POLICIES)
            vals = {p: regret_against(r, p, ref, keep) for p in POLICIES}
            if any(v is None for v in vals.values()):
                continue
            n_scored += 1
            for p in POLICIES:
                per_policy[p].append(vals[p]["mean_fractional_regret"])
        results["by_reference"][label] = summarise(per_policy, label, n_scored)

    # per-cluster: score every valid final cluster, then take the family's best, worst, mean
    for agg in ("best", "worst", "mean"):
        per_policy = {p: [] for p in POLICIES}
        n_scored = 0
        for r in replayed:
            ev = _final_evidence(r)
            if ev is None:
                continue
            refs = refs_per_cluster(ev)
            if not refs:
                continue
            rows = []
            keep = common_declared_cutoffs(r, POLICIES)
            for ref in refs:
                vals = {p: regret_against(r, p, ref, keep) for p in POLICIES}
                if any(v is None for v in vals.values()):
                    continue
                rows.append({p: vals[p]["mean_fractional_regret"] for p in POLICIES})
            if not rows:
                continue
            n_scored += 1
            for p in POLICIES:
                xs = [row[p] for row in rows]
                if agg == "best":
                    per_policy[p].append(min(xs))
                elif agg == "worst":
                    per_policy[p].append(max(xs))
                else:
                    per_policy[p].append(sum(xs) / len(xs))
        results["by_reference"][f"per_cluster_{agg}"] = summarise(
            per_policy, f"per_cluster_{agg}", n_scored)

    # how many valid final clusters each family has, for the write-up
    dist = {}
    for r in replayed:
        ev = _final_evidence(r)
        k = 0 if ev is None else len(ev.valid_sets)
        dist[str(k)] = dist.get(str(k), 0) + 1
    results["valid_final_clusters_distribution"] = dict(sorted(dist.items(),
                                                              key=lambda kv: int(kv[0])))

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINTS / "oracle_sensitivity.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    for label, block in results["by_reference"].items():
        pd = block.get("paired_partitioned_minus_naive")
        if pd:
            print(f"  {label:22s} n={block['n_families_scored']:3d} "
                  f"partitioned-naive {pd['point']:+.3f} "
                  f"[{pd['ci95_lo']:+.3f}, {pd['ci95_hi']:+.3f}]")


if __name__ == "__main__":
    main()

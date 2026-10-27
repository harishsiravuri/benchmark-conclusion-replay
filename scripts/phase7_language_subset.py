"""Phase 7: does the replay result hold on the language-processing families alone?

Zero dollars, deterministic, no model calls. Added 2026-08-13 when the target venue moved
from a web-search-and-data-mining conference to an *ACL venue. The 99 qualifying families
span machine learning broadly, so a reviewer at a computational-linguistics venue will ask
whether the pattern is present in the language families or only in the vision majority.

The partition below is by dataset name, fixed before running anything, and is reported in the
paper so a reader can disagree with it. Three tiers:

  text       natural-language classification, inference, and extraction benchmarks
  kg         knowledge-graph completion benchmarks, a standard *ACL topic
  doc        scene-text and document-understanding benchmarks (optical character recognition)

`language_core` is text + kg. `language_broad` adds doc. Everything else is `other`.

Writes `checkpoints/language_subset.json`.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replaykg.metrics import (  # noqa: E402
    bootstrap_mean, bootstrap_paired_difference, bootstrap_ratio, common_declared_cutoffs,
    family_regret,
    structural_change_decomposition,
)
from replaykg.paths import CHECKPOINTS, DATA, ensure_dirs  # noqa: E402
from replaykg.replay import replay_family  # noqa: E402
from replaykg.substrate import load_families  # noqa: E402
from replaykg.transitions import family_transitions  # noqa: E402

SEED = 13
REPLICATES = 10000
POLICY = "protocol_partitioned"

TEXT = {
    "CR", "MPQA", "MR", "SUBJ", "TREC", "RTE", "XNLI", "NYT", "Rest15", "Rest16",
}
KG = {"FB15k", "WN18", "WN18RR"}
DOC = {
    "ICDAR 2015", "ICDAR2015", "CTW1500", "Total-Text", "MSRA-TD500", "SROIE",
    "IIIT5k", "SVT",
}


def tier(dataset: str) -> str:
    if dataset in TEXT:
        return "text"
    if dataset in KG:
        return "kg"
    if dataset in DOC:
        return "doc"
    return "other"


def main():
    ensure_dirs()
    listing = json.loads((DATA / "replay_families.json").read_text(encoding="utf-8"))
    meta = {f["leaderboard_id"]: f for f in listing["families"]}
    families = [f for f in load_families() if f.leaderboard_id in meta]
    if len(families) != 99:
        raise SystemExit(f"expected 99 qualifying families, matched {len(families)}")

    tiers = collections.Counter()
    members = collections.defaultdict(list)
    for f in families:
        t = tier(meta[f.leaderboard_id]["dataset"])
        tiers[t] += 1
        members[t].append(meta[f.leaderboard_id]["dataset"] + " | "
                          + meta[f.leaderboard_id]["metric"])

    groups = {
        "all_99": lambda t: True,
        "language_core": lambda t: t in ("text", "kg"),
        "language_broad": lambda t: t in ("text", "kg", "doc"),
        "other": lambda t: t == "other",
    }

    out = {
        "generated_by": "scripts/phase7_language_subset.py",
        "note": ("Replay results restricted to the language-processing families. "
                 "Deterministic, zero model cost. Partition by dataset name, fixed "
                 "before running."),
        "seed": SEED, "bootstrap_replicates": REPLICATES,
        "tier_counts": dict(tiers),
        "tier_members": {k: sorted(v) for k, v in members.items()},
        "by_group": {},
    }

    for gname, pred in groups.items():
        sel = [f for f in families if pred(tier(meta[f.leaderboard_id]["dataset"]))]
        replayed = [replay_family(f) for f in sel]

        counts = []
        for r in replayed:
            per_status = collections.Counter()
            for t in family_transitions(r, POLICY):
                per_status[t.status] += 1
            counts.append(dict(per_status))

        total = collections.Counter()
        for c in counts:
            total.update(c)

        regret_pp, regret_naive = [], []
        for r in replayed:
            # same-cutoff basis, as in phase 2: score both policies only on the cutoffs
            # where every policy declared, so the two means cover identical cutoffs
            keep = common_declared_cutoffs(r)
            a = family_regret(r, "protocol_partitioned", "protocol_gated", keep)
            b = family_regret(r, "naive_latest_max", "protocol_gated", keep)
            if a and b:
                regret_pp.append(a["mean_fractional_regret"])
                regret_naive.append(b["mean_fractional_regret"])

        block = {
            "n_families": len(sel),
            "transition_counts": dict(sorted(total.items())),
            "contradicted": total.get("contradicted", 0),
            "decomposition": structural_change_decomposition(counts),
            "n_families_scored_regret": len(regret_pp),
        }
        if regret_pp:
            block["regret_protocol_partitioned"] = bootstrap_mean(
                regret_pp, seed=SEED, replicates=REPLICATES).as_dict()
            block["regret_naive"] = bootstrap_mean(
                regret_naive, seed=SEED, replicates=REPLICATES).as_dict()
            block["paired_partitioned_minus_naive"] = bootstrap_paired_difference(
                list(zip(regret_pp, regret_naive)), seed=SEED,
                replicates=REPLICATES).as_dict()
        out["by_group"][gname] = block

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINTS / "language_subset.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"wrote {path}")
    print("tiers:", dict(tiers))
    for g, b in out["by_group"].items():
        pd = b.get("paired_partitioned_minus_naive")
        line = (f"  {g:16s} families={b['n_families']:3d} contradicted={b['contradicted']}")
        if pd:
            line += (f"  PP-naive {pd['point']:+.3f} "
                     f"[{pd['ci95_lo']:+.3f},{pd['ci95_hi']:+.3f}] n={pd['n_families']}")
        print(line)


if __name__ == "__main__":
    main()

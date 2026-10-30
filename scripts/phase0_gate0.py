"""Phase 0: reproduce Gate 0, run the date robustness join, freeze the family set and split.

Zero dollars. Deterministic and resume safe: rerunning overwrites the same artifacts with
byte-identical content given the same substrate.

Outputs:
  checkpoints/gate0_grid.json      the threshold grid, dating coverage, and the verdict
  checkpoints/dating_robustness.json  the PwC join result
  data/replay_families.json        the frozen qualifying family list, with a content hash
  data/split_map.json              the development / sealed-test assignment
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replaykg.dating_robustness import compare_dating, load_pwc_years  # noqa: E402
from replaykg.gate0 import gate0, qualifying_families  # noqa: E402
from replaykg.paths import (  # noqa: E402
    CHECKPOINTS,
    CLEANED_LEADERBOARDS,
    DATA,
    REPLAY_FAMILIES,
    SPLIT_MAP,
    content_hash,
    ensure_dirs,
)
from replaykg.split import make_split  # noqa: E402
from replaykg.substrate import load_families, verify_substrate  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-pwc-join", action="store_true",
                    help="skip the PwC parquet robustness join (requires pyarrow)")
    args = ap.parse_args()

    ensure_dirs()
    digest = verify_substrate()
    families = list(load_families())

    # --- Gate 0 -----------------------------------------------------------------------
    g0 = gate0(families)
    # Record the file's name and digest, not its absolute path: an absolute path
    # carries the machine account name into the released dataset.
    g0["substrate"] = {"file": CLEANED_LEADERBOARDS.name,
                       "from": "companion comparability resource",
                       "sha256": digest}
    write_json(CHECKPOINTS / "gate0_grid.json", g0)

    qualifying = qualifying_families(families)

    # --- date robustness join ---------------------------------------------------------
    robustness = {"skipped": True}
    if not args.skip_pwc_join:
        wanted = {str(e.arxiv_id) for f in families for e in f.entries}
        pwc_years = load_pwc_years(wanted)
        robustness = compare_dating(families, pwc_years)

        def qualifies_under_pwc(fam, n=10, d=4, s=4) -> bool:
            years = [pwc_years.get(str(e.arxiv_id), e.year) for e in fam.entries]
            years = [y for y in years if y is not None]
            if not years:
                return False
            uniq = sorted(set(years))
            return len(years) >= n and len(uniq) >= d and (uniq[-1] - uniq[0]) >= s

        pwc_ids = {f.leaderboard_id for f in families if qualifies_under_pwc(f)}
        arxiv_ids = {f.leaderboard_id for f in qualifying}
        robustness["qualifying_set_under_pwc_dating"] = len(pwc_ids)
        robustness["qualifying_set_under_arxiv_dating"] = len(arxiv_ids)
        robustness["qualifying_set_identical"] = pwc_ids == arxiv_ids
        robustness["only_arxiv"] = sorted(arxiv_ids - pwc_ids)
        robustness["only_pwc"] = sorted(pwc_ids - arxiv_ids)
    write_json(CHECKPOINTS / "dating_robustness.json", robustness)

    # --- freeze the family set --------------------------------------------------------
    rows = [
        {
            "leaderboard_id": f.leaderboard_id,
            "dataset": f.dataset,
            "metric": f.metric,
            "metric_direction": f.metric_direction,
            "identity_grade": f.identity_grade,
            "n_dated_results": f.n_dated,
            "n_distinct_years": f.n_distinct_years,
            "year_min": f.years[0],
            "year_max": f.years[-1],
            "year_span_elapsed": f.year_span,
            "n_comparable_clusters": f.n_comparable_clusters,
            "multi_protocol": f.is_multi_protocol,
            "static_winner_changed": f.winner_changed,
            "pair_comparable_fraction": f.pair_comparable_fraction,
            "pair_unknown_fraction": f.pair_unknown_fraction,
        }
        for f in sorted(qualifying, key=lambda x: x.leaderboard_id)
    ]
    body = json.dumps(rows, indent=2, sort_keys=False)
    write_json(REPLAY_FAMILIES, {
        "generated_by": "scripts/phase0_gate0.py",
        "substrate_sha256": digest,
        "inclusion_rule": {"min_dated_results": 10, "min_distinct_years": 4,
                           "min_elapsed_year_span": 4},
        "n_families": len(rows),
        "n_multi_protocol": sum(1 for r in rows if r["multi_protocol"]),
        "n_static_winner_changed": sum(1 for r in rows if r["static_winner_changed"]),
        "content_hash_sha256": content_hash(body),
        "families": rows,
    })

    # --- split ------------------------------------------------------------------------
    split = make_split(qualifying)
    split["substrate_sha256"] = digest
    split["family_content_hash_sha256"] = content_hash(body)
    write_json(SPLIT_MAP, split)

    print(json.dumps({
        "gate0_verdict": g0["verdict"],
        "qualifying_families": g0["primary"]["families"],
        "multi_protocol": g0["primary"]["multi_protocol"],
        "static_winner_changed": g0["primary"]["static_winner_changed"],
        "dating_coverage": g0["dating"]["dating_coverage"],
        "pwc_agreement_rate": robustness.get("agreement_rate"),
        "pwc_qualifying_set_identical": robustness.get("qualifying_set_identical"),
        "n_dev": split["n_dev"], "n_test": split["n_test"],
        "family_content_hash": content_hash(body)[:16],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase 1: run the deterministic chronological replay over one split. Zero dollars.

Outputs, all under `data/replay/<split>/`:
  certificates.jsonl   one replay certificate per family, cutoff, and policy
  transitions.jsonl    one record per cutoff-to-cutoff transition per policy
  conclusions.jsonl    the per-cutoff conclusions, for the metrics phase
  manifest.json        substrate hash, noise-model hash, policy parameters, counts

Resume safe: the run is a pure function of the frozen substrate, the frozen family list, and
the policy parameters, so rerunning reproduces the same bytes.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replaykg.certificate import build_replay_certificate, self_consistent  # noqa: E402
from replaykg.noise_adapter import noise_model_provenance  # noqa: E402
from replaykg.paths import (  # noqa: E402
    CLEANED_LEADERBOARDS, DATA, REPLAY_FAMILIES, SPLIT_MAP, content_hash,
)
from replaykg.replay import (  # noqa: E402
    CONSERVATIVE_K, MIN_VALID_SET_SIZE, POLICIES, UNRESOLVED_UNKNOWN_SHARE, replay_family,
)
from replaykg.substrate import load_families, verify_substrate  # noqa: E402
from replaykg.transitions import family_transitions  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    args = ap.parse_args()

    digest = verify_substrate()
    split_map = json.loads(SPLIT_MAP.read_text(encoding="utf-8"))
    frozen = json.loads(REPLAY_FAMILIES.read_text(encoding="utf-8"))
    wanted = {fid for fid, s in split_map["assignment"].items()
              if args.split == "all" or s == args.split}
    if not wanted:
        raise SystemExit(f"no families assigned to split {args.split}")

    families = [f for f in load_families() if f.leaderboard_id in wanted]
    families.sort(key=lambda f: f.leaderboard_id)
    assert len(families) == len(wanted), "split map references a family not in the substrate"

    outdir = DATA / "replay" / args.split
    outdir.mkdir(parents=True, exist_ok=True)

    n_cert = n_trans = n_inconsistent = 0
    with open(outdir / "certificates.jsonl", "w", encoding="utf-8") as fc, \
         open(outdir / "transitions.jsonl", "w", encoding="utf-8") as ft, \
         open(outdir / "conclusions.jsonl", "w", encoding="utf-8") as fk:
        for fam in families:
            replayed = replay_family(fam)
            years = replayed["years"]
            for policy in POLICIES:
                transitions = family_transitions(replayed, policy)
                by_to_cutoff = {t.to_cutoff: t for t in transitions}
                for i, year in enumerate(years):
                    concl = replayed["conclusions"][policy][i]
                    cert = build_replay_certificate(
                        fam, concl, replayed["evidence"][year], by_to_cutoff.get(year))
                    if not self_consistent(cert):
                        n_inconsistent += 1
                    fc.write(json.dumps(cert) + "\n")
                    n_cert += 1
                    row = asdict(concl)
                    row.pop("cluster_winners", None)
                    fk.write(json.dumps(row) + "\n")
                for t in transitions:
                    ft.write(json.dumps(t.as_dict()) + "\n")
                    n_trans += 1

    manifest = {
        "split": args.split,
        "n_families": len(families),
        "n_certificates": n_cert,
        "n_transitions": n_trans,
        "n_self_inconsistent_certificates": n_inconsistent,
        "substrate": {"path": str(CLEANED_LEADERBOARDS), "sha256": digest},
        "family_content_hash_sha256": frozen["content_hash_sha256"],
        "policies": list(POLICIES),
        "policy_parameters": {
            "conservative_k": CONSERVATIVE_K,
            "unresolved_unknown_share": UNRESOLVED_UNKNOWN_SHARE,
            "min_valid_set_size": MIN_VALID_SET_SIZE,
        },
        "noise_model": noise_model_provenance(),
        "spend_usd": 0.0,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["outputs_content_hash"] = content_hash(
        (outdir / "certificates.jsonl").read_text(encoding="utf-8")
        + (outdir / "transitions.jsonl").read_text(encoding="utf-8")
    )
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "noise_model"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

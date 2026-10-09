"""Phase 5: assemble the released dataset. Zero dollars, deterministic.

Writes `dataset/` containing, pointers only and with no excerpt text:

  replay_certificates.jsonl   one certificate per family, cutoff, and policy, over all 99
                              qualifying families
  transitions.jsonl           one record per cutoff-to-cutoff transition per policy
  replay_families.json        the frozen qualifying family list with its content hash
  split_map.json              the development and held-out assignment, seed 13
  propagation_pairs.jsonl     the Phase 3 citation-propagation pairs with their deterministic
                              measures, pointers only
  propagation_llm.jsonl       the model reading of each citing span, all three passes joined
  adjudication_sheet.json     the 47 stratified pairs used for the manual instrument check
  metrics/                    the computed checkpoints reproduced verbatim
  MANIFEST.json               sha256 and row count for every released file
  DATASHEET.md                motivation, composition, collection, uses, and limitations
  REPRODUCE.md                the reproduction map

An excerpt-text guard runs over every released row before the manifest is written, and the
build aborts if any evidence text is found, since the substrate licence permits pointers only.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replaykg.paths import (  # noqa: E402
    CHECKPOINTS, DATA, REPLAY_FAMILIES, REPO_ROOT, SPLIT_MAP, sha256_file,
)

RELEASE = REPO_ROOT / "dataset"

# Keys that would carry redistributable excerpt text. None may appear in a released row.
FORBIDDEN_TEXT_KEYS = {"evidence_quote", "quote", "source_quote", "citing_quote", "excerpt",
                       "abstract", "fulltext", "text"}


class ExcerptTextFound(RuntimeError):
    pass


HOME_PREFIXES = (str(REPO_ROOT.parent), str(Path.home()))


def sanitise(obj):
    """Two release hygiene rules, applied to every value before it is written.

    Absolute filesystem paths are rewritten relative to the project root, so the release does
    not carry the author's home directory. Error strings are reduced to the exception type and
    a short reason, because a parse failure otherwise embeds the raw model response, and a raw
    response could in principle echo span text that the licence does not permit us to
    redistribute.
    """
    if isinstance(obj, dict):
        return {k: (_short_error(v) if k == "error" and isinstance(v, str) else sanitise(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitise(v) for v in obj]
    if isinstance(obj, str):
        out = obj
        for prefix in HOME_PREFIXES:
            out = out.replace(prefix + "/", "").replace(prefix, "")
        return out
    return obj


def _short_error(text: str) -> str:
    """Exception type plus a fixed reason. The raw model response is never released."""
    kind = text.split(":", 1)[0].strip() or "Error"
    if "Could not parse JSON" in text:
        return f"{kind}: response was not parseable as JSON (raw response not released)"
    return f"{kind}: call failed (detail not released)"


def assert_no_excerpt_text(obj, where: str) -> None:
    """The licence permits pointers only, so this guard is a release blocker, not a warning."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN_TEXT_KEYS and isinstance(v, str) and v.strip():
                raise ExcerptTextFound(f"{where}: key '{k}' carries excerpt text")
            assert_no_excerpt_text(v, where)
    elif isinstance(obj, list):
        for v in obj:
            assert_no_excerpt_text(v, where)


def copy_jsonl(src: Path, dst: Path, strip_keys=()) -> int:
    n = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            for k in strip_keys:
                row.pop(k, None)
            row = sanitise(row)
            assert_no_excerpt_text(row, dst.name)
            fout.write(json.dumps(row) + "\n")
            n += 1
    return n


def copy_json(src: Path, dst: Path) -> int:
    obj = sanitise(json.loads(src.read_text(encoding="utf-8")))
    assert_no_excerpt_text(obj, dst.name)
    dst.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    return 1


def join_llm_passes(dst: Path) -> int:
    """One row per pair, carrying all three model passes and no span text."""
    src = DATA / "propagation"
    passes = {}
    for name in ("extract", "same_quantity", "extract_agreement"):
        path = src / f"llm_{name}.jsonl"
        passes[name] = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    pairs = [json.loads(l) for l in open(src / "pairs.jsonl", encoding="utf-8") if l.strip()]
    n = 0
    with open(dst, "w", encoding="utf-8") as fh:
        for i, pair in enumerate(pairs):
            row = {
                "cell": pair["cell"],
                "source_paper_id": pair["source"]["paper_id"],
                "citing_paper_id": pair["citing"]["paper_id"],
                "extract": {k: v for k, v in passes["extract"][i].items()
                            if k not in ("cell", "source_paper", "citing_paper")},
                "same_quantity": {k: v for k, v in passes["same_quantity"][i].items()
                                  if k not in ("cell", "source_paper", "citing_paper")},
                "extract_second_backbone": {
                    k: v for k, v in passes["extract_agreement"][i].items()
                    if k not in ("cell", "source_paper", "citing_paper")},
            }
            row = sanitise(row)
            assert_no_excerpt_text(row, dst.name)
            fh.write(json.dumps(row) + "\n")
            n += 1
    return n


def strip_adjudication_sheet(src: Path, dst: Path) -> int:
    """The sheet carries spans for manual reading, so the released copy drops them."""
    obj = json.loads(src.read_text(encoding="utf-8"))
    for item in obj["items"]:
        for side in ("source", "citing"):
            item[side].pop("quote", None)
        item["quote_redacted"] = True
    obj = sanitise(obj)
    obj["note"] = ("Spans are removed from the released copy because the substrate licence "
                   "permits pointers only. Consult the cited arXiv paper directly.")
    assert_no_excerpt_text(obj, dst.name)
    dst.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    return len(obj["items"])


def main() -> int:
    if RELEASE.exists():
        shutil.rmtree(RELEASE)
    (RELEASE / "metrics").mkdir(parents=True)

    counts = {}
    counts["replay_certificates.jsonl"] = copy_jsonl(
        DATA / "replay" / "all" / "certificates.jsonl", RELEASE / "replay_certificates.jsonl")
    counts["transitions.jsonl"] = copy_jsonl(
        DATA / "replay" / "all" / "transitions.jsonl", RELEASE / "transitions.jsonl")
    counts["replay_families.json"] = copy_json(REPLAY_FAMILIES, RELEASE / "replay_families.json")
    counts["split_map.json"] = copy_json(SPLIT_MAP, RELEASE / "split_map.json")
    counts["propagation_pairs.jsonl"] = copy_jsonl(
        DATA / "propagation" / "pairs.jsonl", RELEASE / "propagation_pairs.jsonl")
    counts["propagation_llm.jsonl"] = join_llm_passes(RELEASE / "propagation_llm.jsonl")
    counts["adjudication_sheet.json"] = strip_adjudication_sheet(
        DATA / "propagation" / "adjudication_sheet.json", RELEASE / "adjudication_sheet.json")

    for name in ("gate0_grid.json", "dating_robustness.json", "metrics_all.json",
                 "metrics_dev.json", "metrics_test.json", "sensitivity_all.json",
                 "sensitivity_dev.json", "band_check_all.json", "propagation_census.json",
                 "propagation_llm_analysis.json", "phase3_cost_projection.json"):
        src = CHECKPOINTS / name
        if src.exists():
            counts[f"metrics/{name}"] = copy_json(src, RELEASE / "metrics" / name)

    shutil.copy(REPO_ROOT / "data" / "propagation" / "adjudication_verdicts.md",
                RELEASE / "adjudication_verdicts.md")
    counts["adjudication_verdicts.md"] = 1
    # the datasheet is authored under docs/ and the reproduction map at the repository root;
    # both are copied in so MANIFEST.json pins every file the release actually ships
    shutil.copy(REPO_ROOT / "docs" / "DATASHEET.md", RELEASE / "DATASHEET.md")
    counts["DATASHEET.md"] = 1
    shutil.copy(REPO_ROOT / "REPRODUCE.md", RELEASE / "REPRODUCE.md")
    counts["REPRODUCE.md"] = 1

    files = []
    for path in sorted(RELEASE.rglob("*")):
        if path.is_file():
            rel = str(path.relative_to(RELEASE))
            files.append({"path": rel, "sha256": sha256_file(path),
                          "size_bytes": path.stat().st_size,
                          "rows": counts.get(rel)})

    manifest = {
        "resource": "replaykg: chronological replay of protocol-qualified benchmark conclusions",
        "version": "1.0",
        "license": ("CC-BY-SA 4.0, matching the upstream comparability resource it is derived from. "
                    "Pointers only, no excerpt text."),
        "derived_from": {
            "substrate": "upstream/data/cleaned_leaderboards/cleaned_leaderboards.jsonl",
            "substrate_sha256":
                "b2dd904793a6f5dac0815a2936553d67aac2005e2dbb1cdc735fd79073ef2472",
            "snapshot": "Papers-with-Code archive frozen 2025-07-28",
            "extraction": "upstream/data/extractions/tuples.parquet (Phase 3 only)",
        },
        "evidence_policy": "POINTERS ONLY (arxiv_id, arxiv_abs_url, source_location); "
                           "NO excerpt text. Enforced by a guard in scripts/phase5_release.py.",
        "seed": 13,
        "files": files,
    }
    (RELEASE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                           encoding="utf-8")
    print(json.dumps({"n_files": len(files),
                      "rows": {k: v for k, v in counts.items() if v and v > 1}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

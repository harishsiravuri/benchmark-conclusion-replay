"""Phase 3: numeric citation propagation. Phase 3a is free; Phase 3b spends.

Run order:
  --stage census    deterministic census over every propagation pair (zero dollars)
  --stage project   print the cost projection for the language-model passes and stop
  --stage llm       run the two language-model passes, cached and cost-logged
  --stage analyse   combine the deterministic census with the model reading

The 25 dollar Phase 3 sub-cap is enforced in `replaykg.common.costlog`, which refuses any
non-cached call once cumulative spend reaches the cap. The projection is printed and recorded
before the first call, and the run aborts if the projection exceeds the sub-cap.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from replaykg.common import costlog  # noqa: E402
from replaykg.common.llm import CachedLLM, parse_json_response  # noqa: E402
from replaykg.metrics import bootstrap_ratio  # noqa: E402
from replaykg.paths import CHECKPOINTS, CONFIG, DATA, REPLAY_FAMILIES, ensure_dirs  # noqa: E402
from replaykg.propagation import build_pairs  # noqa: E402
from replaykg.propagation_llm import (  # noqa: E402
    ADJUDICATE_SYSTEM, EXTRACT_SYSTEM, adjudicate_user, extract_user, project_cost,
)

OUT = DATA / "propagation"
SUB_CAP = 25.0
CONCURRENCY = 8


def load_models() -> dict:
    return yaml.safe_load((CONFIG / "models.yaml").read_text(encoding="utf-8"))


def replay_family_papers() -> set[str]:
    """arXiv identifiers appearing in the 99 frozen replay families."""
    import sys as _sys
    from replaykg.substrate import load_families
    frozen = json.loads(REPLAY_FAMILIES.read_text(encoding="utf-8"))
    ids = {r["leaderboard_id"] for r in frozen["families"]}
    papers = set()
    for fam in load_families():
        if fam.leaderboard_id in ids:
            papers.update(e.arxiv_id for e in fam.entries)
    del _sys
    return papers


# ---------------------------------------------------------------------------------------
# Phase 3a: deterministic census
# ---------------------------------------------------------------------------------------

def census(pairs) -> dict:
    """Pooled rates with family-of-cell bootstrap confidence intervals.

    The resampling unit is the canonical cell, because several citing papers may repeat the
    same source number and those repetitions are not independent observations.
    """
    by_cell = collections.defaultdict(list)
    for p in pairs:
        by_cell[p.cell].append(p)

    def rate(predicate, subset=None):
        units = []
        for cell, ps in by_cell.items():
            sel = [p for p in ps if subset is None or subset(p)]
            if sel:
                units.append((sum(1 for p in sel if predicate(p)), len(sel)))
        return bootstrap_ratio(units).as_dict()

    val_source = lambda p: p.source_split in ("val", "dev", "validation")  # noqa: E731

    return {
        "n_pairs": len(pairs),
        "n_cells": len(by_cell),
        "n_source_papers": len({p.source_paper for p in pairs}),
        "n_citing_papers": len({p.citing_paper for p in pairs}),
        "copy_fidelity": {
            "exact": rate(lambda p: p.copy_fidelity == "exact"),
            "exact_or_rounding": rate(lambda p: p.copy_fidelity in ("exact", "rounding")),
            "mismatch": rate(lambda p: p.copy_fidelity == "mismatch"),
            "counts": dict(collections.Counter(p.copy_fidelity for p in pairs)),
        },
        "qualifier_preservation": {
            "all_observed_facets_preserved": rate(lambda p: p.qualifier_preserved),
            "split_dropped": rate(lambda p: p.facet_relations["split"] == "dropped"),
            "split_changed": rate(lambda p: p.facet_relations["split"] == "changed"),
            "facet_counts": {
                f: dict(collections.Counter(p.facet_relations[f] for p in pairs))
                for f in ("split", "unit", "metric_surface")
            },
        },
        "validation_cited_as_test": {
            "among_validation_sources": rate(lambda p: p.validation_cited_as_test, val_source),
            "qualifier_lost_among_validation_sources": rate(
                lambda p: p.validation_qualifier_lost, val_source),
            "n_validation_source_pairs": sum(1 for p in pairs if val_source(p)),
            "n_validation_cited_as_test": sum(1 for p in pairs if p.validation_cited_as_test),
        },
        "source_split_distribution": dict(collections.Counter(p.source_split for p in pairs)),
        "citation_lag_years": dict(collections.Counter(p.citing_year - p.source_year for p in pairs)),
    }


# ---------------------------------------------------------------------------------------
# Phase 3b: language-model passes
# ---------------------------------------------------------------------------------------

async def run_pass(llm, pairs, system, user_fn, cfg, stage) -> list[dict]:
    sem = asyncio.Semaphore(CONCURRENCY)
    results: list[dict] = [None] * len(pairs)

    async def one(i, p):
        async with sem:
            try:
                r = await llm.complete(
                    model_id=cfg["model_id"], system=system, user=user_fn(p),
                    temperature=cfg["temperature"], max_tokens=cfg["max_tokens"],
                    stage=stage, price_in_per_m=cfg["price_in_per_m"],
                    price_out_per_m=cfg["price_out_per_m"])
                parsed = parse_json_response(r.text)
                results[i] = {"ok": True, "cached": r.cached, "cost_usd": r.cost_usd, **parsed}
            except Exception as exc:  # recorded, never silently dropped
                results[i] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}

    await asyncio.gather(*(one(i, p) for i, p in enumerate(pairs)))
    return results


async def llm_stage(pairs, models) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    llm = CachedLLM(budget_cap_usd=SUB_CAP)
    try:
        for name, system, user_fn in (
            ("extract", EXTRACT_SYSTEM, extract_user),
            ("same_quantity", ADJUDICATE_SYSTEM, adjudicate_user),
            ("extract_agreement", EXTRACT_SYSTEM, extract_user),
        ):
            cfg = models[f"propagation_{name}"]
            print(f"[{name}] model {cfg['model_id']} over {len(pairs)} pairs")
            res = await run_pass(llm, pairs, system, user_fn, cfg, f"phase3_{name}")
            path = OUT / f"llm_{name}.jsonl"
            with open(path, "w", encoding="utf-8") as fh:
                for p, r in zip(pairs, res):
                    fh.write(json.dumps({
                        "cell": list(p.cell), "source_paper": p.source_paper,
                        "citing_paper": p.citing_paper, "model": cfg["model_id"], **r}) + "\n")
            spent = costlog.cumulative_spend()
            ok = sum(1 for r in res if r.get("ok"))
            print(f"[{name}] wrote {path}, {ok}/{len(res)} ok, cumulative spend "
                  f"${spent:.4f} of ${SUB_CAP:.2f}")
    finally:
        await llm.aclose()


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def analyse(pairs) -> int:
    """Combine the deterministic census with the model reading of the citing spans."""
    extract = _load_jsonl(OUT / "llm_extract.jsonl")
    same = _load_jsonl(OUT / "llm_same_quantity.jsonl")
    agreement_path = OUT / "llm_extract_agreement.jsonl"
    # The second-backbone pass is a robustness check, not a load-bearing input. It is written
    # only when its own pass completes, so the analysis runs without it and records that fact.
    agree = _load_jsonl(agreement_path) if agreement_path.exists() else [
        {"ok": False, "error": "second-backbone pass not available"} for _ in pairs]
    have_agreement = agreement_path.exists()
    assert len(extract) == len(same) == len(agree) == len(pairs), "pass lengths disagree"

    by_cell = collections.defaultdict(list)
    rows = []
    for p, e, sq, ag in zip(pairs, extract, same, agree):
        row = {"pair": p, "extract": e, "same": sq, "agreement": ag}
        rows.append(row)
        by_cell[p.cell].append(row)

    def rate(predicate, subset=None):
        units = []
        for cell, rs in by_cell.items():
            sel = [r for r in rs if subset is None or subset(r)]
            if sel:
                units.append((sum(1 for r in sel if predicate(r)), len(sel)))
        return bootstrap_ratio(units).as_dict()

    ok = lambda r: r["extract"].get("ok") and r["same"].get("ok")  # noqa: E731
    same_quantity = lambda r: ok(r) and r["same"].get("relation") == "same_quantity"  # noqa: E731
    val_source = lambda r: r["pair"].source_split in ("val", "dev", "validation")  # noqa: E731

    # extraction error against the deterministic reading of the same span
    def value_agrees(r):
        v = r["extract"].get("cited_value")
        return v is not None and abs(float(v) - r["pair"].citing_value) <= 1e-6

    def split_agrees(r):
        stated = r["extract"].get("stated_split")
        return (stated or None) == (r["pair"].citing_split or None)

    def backbone_agrees(r):
        a, b = r["extract"].get("stated_split"), r["agreement"].get("stated_split")
        return (a or None) == (b or None)

    def explicit(r):
        return bool(r["extract"].get("split_is_explicit"))

    out = {
        "spend_usd": costlog.cumulative_spend(),
        "sub_cap_usd": SUB_CAP,
        "n_pairs": len(pairs),
        "n_cells": len(by_cell),
        "second_backbone_pass_available": have_agreement,
        "pass_success": {
            "extract_ok": sum(1 for r in rows if r["extract"].get("ok")),
            "same_quantity_ok": sum(1 for r in rows if r["same"].get("ok")),
            "agreement_ok": sum(1 for r in rows if r["agreement"].get("ok")),
        },
        "instrument_check": {
            "note": ("agreement between the model reading of a citing span and the frozen "
                     "upstream extraction of the same span; disagreement bounds how much of "
                     "the census reflects extraction error rather than citation behaviour"),
            "model_value_matches_extracted_value": rate(value_agrees, ok),
            "model_split_matches_extracted_split": rate(split_agrees, ok),
            "two_backbones_agree_on_stated_split": rate(
                backbone_agrees, lambda r: r["extract"].get("ok") and r["agreement"].get("ok")),
        },
        "same_quantity_adjudication": {
            "relation_counts": dict(collections.Counter(
                r["same"].get("relation") for r in rows if r["same"].get("ok"))),
            "share_same_quantity": rate(same_quantity, ok),
            "share_different_configuration": rate(
                lambda r: r["same"].get("relation") == "different_configuration", ok),
        },
        "copy_fidelity_adjudicated": {
            "note": ("copy fidelity restricted to pairs the adjudicator judges to be the same "
                     "quantity; this is the honest denominator for a claim about copying"),
            "exact_or_rounding_among_same_quantity": rate(
                lambda r: r["pair"].copy_fidelity in ("exact", "rounding"), same_quantity),
            "mismatch_among_same_quantity": rate(
                lambda r: r["pair"].copy_fidelity == "mismatch", same_quantity),
            "n_same_quantity_pairs": sum(1 for r in rows if same_quantity(r)),
            "mismatch_among_different_configuration": rate(
                lambda r: r["pair"].copy_fidelity == "mismatch",
                lambda r: r["same"].get("relation") == "different_configuration"),
        },
        "qualifier_statement_in_the_citing_span": {
            "note": ("measured by the model on the span that actually carries the number; a "
                     "span that omits the split is a span a reader can consume without ever "
                     "seeing the qualifier, which is the phenomenon of interest"),
            "split_explicitly_stated": rate(explicit, ok),
            "split_explicitly_stated_among_same_quantity": rate(explicit, same_quantity),
            "split_explicitly_stated_when_source_is_validation": rate(
                explicit, lambda r: ok(r) and val_source(r)),
        },
        "validation_cited_as_test": {
            "n_validation_source_pairs": sum(1 for r in rows if val_source(r)),
            "model_states_test_when_source_is_validation": rate(
                lambda r: (r["extract"].get("stated_split") == "test"),
                lambda r: ok(r) and val_source(r)),
            "qualifier_absent_when_source_is_validation": rate(
                lambda r: not explicit(r), lambda r: ok(r) and val_source(r)),
            "deterministic_validation_cited_as_test": rate(
                lambda r: r["pair"].validation_cited_as_test, val_source),
        },
    }

    (CHECKPOINTS / "propagation_llm_analysis.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")

    # author-adjudication sheet: a stratified subset for manual checking
    import random
    rng = random.Random(13)
    strata = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if not ok(r):
            continue
        strata[(r["same"].get("relation"), r["pair"].copy_fidelity)].append(i)
    sheet = []
    for key in sorted(strata, key=lambda k: str(k)):
        idxs = sorted(strata[key])
        rng.shuffle(idxs)
        for i in idxs[:6]:
            r = rows[i]
            sheet.append({
                "index": i, "stratum": list(key),
                "cell": {"method": r["pair"].cell_names[0], "dataset": r["pair"].cell_names[1],
                         "metric": r["pair"].cell_names[2]},
                "source": {"paper": r["pair"].source_paper, "value": r["pair"].source_value,
                           "split": r["pair"].source_split,
                           "quote": (r["pair"].source_quote or "")[:400]},
                "citing": {"paper": r["pair"].citing_paper, "value": r["pair"].citing_value,
                           "split": r["pair"].citing_split,
                           "quote": (r["pair"].citing_quote or "")[:400]},
                "model_extract": {k: r["extract"].get(k) for k in
                                  ("cited_value", "stated_split", "split_is_explicit",
                                   "other_stated_conditions")},
                "model_relation": {k: r["same"].get(k) for k in
                                   ("relation", "differing_condition", "rationale")},
                "author_verdict": None,
            })
    (OUT / "adjudication_sheet.json").write_text(
        json.dumps({"n": len(sheet), "seed": 13, "items": sheet}, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("instrument_check",)}, indent=2)[:3000])
    print(f"adjudication sheet: {len(sheet)} items")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["census", "project", "llm", "analyse"], required=True)
    args = ap.parse_args()
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs()
    models = load_models()

    if args.stage == "census":
        replay_papers = replay_family_papers()
        in_replay = [p for p in pairs
                     if p.source_paper in replay_papers or p.citing_paper in replay_papers]
        out = {
            "spend_usd": 0.0,
            "definition": (
                "a propagation pair is one own-result reading of a canonical cell in the "
                "earliest paper that reports it, together with one quoted reading of the "
                "same cell in a later, different paper"
            ),
            "full_corpus": census(pairs),
            "intersecting_the_99_replay_families": census(in_replay) if in_replay else None,
        }
        (CHECKPOINTS / "propagation_census.json").write_text(
            json.dumps(out, indent=2) + "\n", encoding="utf-8")
        with open(OUT / "pairs.jsonl", "w", encoding="utf-8") as fh:
            for p in pairs:
                fh.write(json.dumps(p.as_dict()) + "\n")
        print(json.dumps({k: v for k, v in out["full_corpus"].items()
                          if k not in ("citation_lag_years",)}, indent=2)[:2600])
        return 0

    if args.stage == "project":
        total = 0.0
        rows = []
        for name, system, user_fn in (("extract", EXTRACT_SYSTEM, extract_user),
                                      ("same_quantity", ADJUDICATE_SYSTEM, adjudicate_user),
                                      ("extract_agreement", EXTRACT_SYSTEM, extract_user)):
            cfg = models[f"propagation_{name}"]
            pr = project_cost(pairs, system, user_fn,
                              cfg["price_in_per_m"], cfg["price_out_per_m"])
            pr["stage"] = name
            pr["model_id"] = cfg["model_id"]
            rows.append(pr)
            total += pr["projected_usd"]
        payload = {"sub_cap_usd": SUB_CAP, "projected_total_usd": total, "by_stage": rows,
                   "within_sub_cap": total <= SUB_CAP,
                   "cumulative_spend_before": costlog.cumulative_spend()}
        (CHECKPOINTS / "phase3_cost_projection.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0 if total <= SUB_CAP else 1

    if args.stage == "llm":
        proj = json.loads((CHECKPOINTS / "phase3_cost_projection.json").read_text())
        if not proj["within_sub_cap"]:
            raise SystemExit("projection exceeds the Phase 3 sub-cap; not running")
        asyncio.run(llm_stage(pairs, models))
        return 0

    return analyse(pairs)


if __name__ == "__main__":
    raise SystemExit(main())

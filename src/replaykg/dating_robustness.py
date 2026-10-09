"""Date robustness join against the frozen Papers-with-Code papers table.

The primary dating rule is deterministic and needs no external join: every substrate entry
carries an arXiv identifier whose YYMM prefix yields the publication year. This module runs
the robustness check the study protocol requires, in two parts.

1. Any identifier that the arXiv rule cannot resolve is looked up in the frozen PwC papers
   table and dated from its `date` field. On the frozen substrate there are zero such
   identifiers, so this branch is a guard rather than a correction.
2. For every identifier the arXiv rule does resolve, the resolved year is compared against
   the PwC `date` year. Disagreement is reported, and the qualifying family set is recomputed
   under PwC dating so the Gate 0 conclusion can be confirmed rather than assumed.

The PwC table is read only. The join reads two columns and nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path

from replaykg.paths import PWC_PAPERS_MANIFEST


def pwc_parquet_files(manifest_path: Path = PWC_PAPERS_MANIFEST) -> list[Path]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return [Path(f["path"]) for f in manifest["files"]]


def load_pwc_years(wanted: set[str] | None = None,
                   manifest_path: Path = PWC_PAPERS_MANIFEST) -> dict[str, int]:
    """Map arXiv identifier to publication year from the frozen PwC `date` column."""
    import pyarrow.parquet as pq

    out: dict[str, int] = {}
    for path in pwc_parquet_files(manifest_path):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=65536, columns=["arxiv_id", "date"]):
            ids = batch.column("arxiv_id").to_pylist()
            dates = batch.column("date").to_pylist()
            for aid, dt in zip(ids, dates):
                if not aid or dt is None:
                    continue
                aid = str(aid).strip()
                if wanted is not None and aid not in wanted:
                    continue
                out.setdefault(aid, dt.year)
    return out


def compare_dating(families, pwc_years: dict[str, int]) -> dict:
    """Agreement between arXiv-prefix dating and PwC-table dating, over all entries."""
    matched = agree = 0
    disagreements: list[dict] = []
    unresolved_by_arxiv: list[str] = []
    for fam in families:
        for e in fam.entries:
            if e.year is None:
                unresolved_by_arxiv.append(e.arxiv_id)
            py = pwc_years.get(str(e.arxiv_id))
            if py is None:
                continue
            matched += 1
            if py == e.year:
                agree += 1
            elif len(disagreements) < 200:
                disagreements.append(
                    {"arxiv_id": e.arxiv_id, "arxiv_prefix_year": e.year, "pwc_year": py}
                )
    return {
        "entries_joined_to_pwc": matched,
        "years_agree": agree,
        "agreement_rate": (agree / matched) if matched else None,
        "n_disagreements": matched - agree,
        "disagreement_sample": disagreements[:50],
        "n_unresolved_by_arxiv_rule": len(unresolved_by_arxiv),
    }

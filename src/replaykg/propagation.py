"""Phase 3a: deterministic census of numeric citation propagation. Zero dollars.

The frozen upstream extraction (`upstream/data/extractions/tuples.parquet`) marks each
reported tuple with `is_own_result`, that is whether the paper is reporting its own number or
quoting somebody else's. A pair consisting of one own-result reading in paper P and one quoted
reading of the same canonical (method, dataset, metric) cell in a later paper Q is a numeric
citation: Q is repeating P's number.

Three quantities are measured on those pairs, following Greenberg (BMJ, 2009) on citation
distortion and the quotation-error literature, with the delta that the object here is a
numeric result cell and its protocol qualifier rather than a textual claim.

  copy fidelity                 does the quoted number equal the source number, after the
                                deterministic percent-versus-fraction reconciliation;
  qualifier preservation        is the source protocol qualifier retained, dropped, changed,
                                or added, per facet;
  validation-cited-as-test      the specific distortion in which a source reading on a
                                validation or development split is quoted as a test result.

The source is taken to be the EARLIEST own-result reading of the cell, because that is the
originating paper, and a quoted reading counts only if its paper is dated at or after the
source. No language model is used here.
"""

from __future__ import annotations

import collections
import importlib.util
import sys
from dataclasses import dataclass
from functools import lru_cache

from replaykg.paths import PAPER32_ROOT, PAPER32_SRC
from replaykg.substrate import arxiv_year

TUPLES = PAPER32_ROOT / "data" / "extractions" / "tuples.parquet"

# Facets a citing paper may or may not carry over from the source.
FACETS = ("split", "unit", "metric_surface")
VALIDATION_FAMILIES = {"val", "dev", "validation"}


@lru_cache(maxsize=1)
def _ch3_rules():
    """The the upstream resource split and surface normalisers, imported read-only, not copied."""
    if str(PAPER32_SRC) not in sys.path:
        sys.path.insert(0, str(PAPER32_SRC))
    spec = importlib.util.spec_from_file_location(
        "ch3_census_surface", PAPER32_SRC / "census" / "surface.py")
    surface = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = surface
    spec.loader.exec_module(surface)
    spec2 = importlib.util.spec_from_file_location("ch3_judge_rules", PAPER32_SRC / "judge" / "rules.py")
    rules = importlib.util.module_from_spec(spec2)
    sys.modules[spec2.name] = rules
    spec2.loader.exec_module(rules)
    return surface, rules


def split_family(raw):
    _, rules = _ch3_rules()
    return rules._split_family(raw)


def norm_surface(raw):
    surface, _ = _ch3_rules()
    return surface._norm(raw) or None


def reconcile(a: float, b: float) -> tuple[float, float, str]:
    """Percent-versus-fraction reconciliation, the only normalisation applied.

    Returns the two values on a common scale and the operation applied. A value at or below
    1.5 beside a value above 1.5 is read as a fraction beside a percentage.
    """
    if a is None or b is None:
        return a, b, "none"
    if abs(a) <= 1.5 and abs(b) > 1.5:
        return a * 100.0, b, "left_fraction_to_percent"
    if abs(b) <= 1.5 and abs(a) > 1.5:
        return a, b * 100.0, "right_fraction_to_percent"
    return a, b, "none"


def facet_relation(source, citing) -> str:
    if source is None and citing is None:
        return "both_missing"
    if source is not None and citing is None:
        return "dropped"
    if source is None and citing is not None:
        return "added"
    return "preserved" if source == citing else "changed"


@dataclass
class PropagationPair:
    cell: tuple                      # canonical (method_id, dataset_id, metric_id)
    cell_names: tuple                # human-readable (method, dataset, metric) as reported
    source_paper: str
    source_year: int
    source_value: float
    source_split: str | None
    source_unit: str | None
    source_surface: str | None
    source_quote: str
    citing_paper: str
    citing_year: int
    citing_value: float
    citing_split: str | None
    citing_unit: str | None
    citing_surface: str | None
    citing_quote: str
    citing_input_kind: str

    # --- deterministic measures -------------------------------------------------------
    @property
    def reconciled(self):
        return reconcile(self.source_value, self.citing_value)

    @property
    def abs_gap(self) -> float:
        a, b, _ = self.reconciled
        return abs(a - b)

    @property
    def copy_fidelity(self) -> str:
        """`exact`, `rounding` (within 0.05 on the reconciled scale), or `mismatch`."""
        gap = self.abs_gap
        if gap == 0:
            return "exact"
        return "rounding" if gap <= 0.05 else "mismatch"

    @property
    def facet_relations(self) -> dict:
        return {
            "split": facet_relation(self.source_split, self.citing_split),
            "unit": facet_relation(self.source_unit, self.citing_unit),
            "metric_surface": facet_relation(self.source_surface, self.citing_surface),
        }

    @property
    def qualifier_preserved(self) -> bool:
        """True when no observed source facet is dropped or changed by the citing paper."""
        return all(r in ("preserved", "both_missing", "added")
                   for r in self.facet_relations.values())

    @property
    def validation_cited_as_test(self) -> bool:
        return (self.source_split in VALIDATION_FAMILIES
                and self.citing_split == "test")

    @property
    def validation_qualifier_lost(self) -> bool:
        """Source is a validation or development reading and the citing paper does not say so."""
        return (self.source_split in VALIDATION_FAMILIES
                and self.citing_split != self.source_split)

    def as_dict(self) -> dict:
        a, b, op = self.reconciled
        return {
            "cell": {"method_id": self.cell[0], "dataset_id": self.cell[1],
                     "metric_id": self.cell[2],
                     "method": self.cell_names[0], "dataset": self.cell_names[1],
                     "metric": self.cell_names[2]},
            "source": {"paper_id": self.source_paper, "year": self.source_year,
                       "value": self.source_value, "split": self.source_split,
                       "unit": self.source_unit, "metric_surface": self.source_surface},
            "citing": {"paper_id": self.citing_paper, "year": self.citing_year,
                       "value": self.citing_value, "split": self.citing_split,
                       "unit": self.citing_unit, "metric_surface": self.citing_surface,
                       "input_kind": self.citing_input_kind},
            "reconciliation": op,
            "reconciled_values": [a, b],
            "abs_gap": self.abs_gap,
            "copy_fidelity": self.copy_fidelity,
            "facet_relations": self.facet_relations,
            "qualifier_preserved": self.qualifier_preserved,
            "validation_cited_as_test": self.validation_cited_as_test,
            "validation_qualifier_lost": self.validation_qualifier_lost,
        }


def load_tuples() -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(TUPLES).to_pylist()


def build_pairs(rows=None) -> list[PropagationPair]:
    """All source-to-citing numeric propagation pairs in the frozen extraction."""
    rows = load_tuples() if rows is None else rows
    cells: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in rows:
        if not (r["method_id"] and r["dataset_id"] and r["metric_id"]):
            continue
        if r["value"] is None:
            continue
        if arxiv_year(r["paper_id"]) is None:
            continue
        cells[(r["method_id"], r["dataset_id"], r["metric_id"])].append(r)

    pairs: list[PropagationPair] = []
    for cell, rs in sorted(cells.items()):
        own = [r for r in rs if r["is_own_result"]]
        if not own:
            continue
        own.sort(key=lambda r: (arxiv_year(r["paper_id"]), r["paper_id"]))
        src = own[0]
        src_year = arxiv_year(src["paper_id"])
        for r in rs:
            if r["is_own_result"] or r["paper_id"] == src["paper_id"]:
                continue
            yr = arxiv_year(r["paper_id"])
            if yr < src_year:
                continue
            pairs.append(PropagationPair(
                cell=cell,
                cell_names=(src.get("method") or r.get("method"),
                            src.get("dataset") or r.get("dataset"),
                            src.get("metric") or r.get("metric")),
                source_paper=src["paper_id"], source_year=src_year,
                source_value=src["value"], source_split=split_family(src["split"]),
                source_unit=norm_surface(src["unit"]), source_surface=norm_surface(src["metric"]),
                source_quote=src.get("evidence_quote") or "",
                citing_paper=r["paper_id"], citing_year=yr,
                citing_value=r["value"], citing_split=split_family(r["split"]),
                citing_unit=norm_surface(r["unit"]), citing_surface=norm_surface(r["metric"]),
                citing_quote=r.get("evidence_quote") or "",
                citing_input_kind=r.get("input_kind") or "",
            ))
    return pairs

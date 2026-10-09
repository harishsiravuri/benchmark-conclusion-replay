# Datasheet: replaykg, a chronological replay of protocol-qualified benchmark conclusions

Version 1.0. Licence CC-BY-SA 4.0, matching the upstream comparability resource this resource is derived
from. Pointers only, no excerpt text. Seed 13 throughout.

House style applies: no contractions, no em dashes, hedged claims, bootstrap 95 percent
confidence intervals reported with n and the test named inline.

## Motivation

### Why was this dataset created?

To make the temporal stability of benchmark conclusions measurable rather than assumed. A
static leaderboard shows what is believed now. It cannot show what was justified in 2019, how
that conclusion ended, or whether it ended because somebody did better work or because the
comparison stopped being valid. This resource records, for 99 benchmark families, what the
justified comparative conclusion was at every yearly cutoff under four update policies, how
each conclusion ended, and what that cost in retrospect.

It also records a second measurement, on how numeric results propagate through citations and
whether the protocol qualifier travels with them.

### Who created it and who funded it?

Created for this study of a solo-authored doctoral project on protocol-qualified scientific
evidence. No external funding is associated with this resource.

## Composition

### What do the instances represent?

Four kinds of instance.

1. **Replay certificates**, 2,580 rows in `replay_certificates.jsonl`. One row per (benchmark
   family, cutoff year, update policy). Each records the declared winner, the valid comparison
   set and its size, the unknown-comparability mass, the protocol signature of the principal
   partition, the answer status, the margin over the runner-up and whether it clears the noise
   band, the transition status into that cutoff, and pointer-only provenance for the winning
   entry.
2. **Transitions**, 2,184 rows in `transitions.jsonl`. One row per cutoff-to-cutoff transition
   per policy, carrying the ordered classifier's status and its stated reason.
3. **Citation-propagation pairs**, 807 rows in `propagation_pairs.jsonl`, with the model
   readings joined in `propagation_llm.jsonl`. One row per (source reading, later quoting
   reading) of the same canonical (method, dataset, metric) cell. Note that 611 rows in
   `propagation_llm.jsonl` carry a `same_quantity.rationale` field. These are the model's own
   paraphrased justifications, not quotations: they are machine-generated text, and they
   contain no passage copied from a source paper. The pointers-only policy is unaffected.
4. **Supporting records**: the frozen family list, the split map, the manual adjudication
   sheet and its written verdicts, and every computed metrics checkpoint.

### How many instances, and of what?

99 benchmark families, of which 45 are multi-protocol and 2 report a metric whose direction the
upstream resource could not determine. 4,879 dated entries are visible at the final cutoff.
546 transitions under the primary protocol-partitioned policy. 807 propagation pairs over 370
canonical cells, 162 source papers, and 240 citing papers.

### Is it a sample or the complete set?

The 99 families are the complete set of families in the 4,438-family substrate that satisfy the
inclusion rule fixed before counting: at least 10 dated results, in at least 4 distinct
publication years, spanning at least 4 elapsed calendar years. The full threshold grid is
released in `metrics/gate0_grid.json` so the sensitivity of that rule is auditable.

The 807 propagation pairs are the complete set derivable from the frozen upstream extraction,
not a sample.

### What data does each instance consist of?

Canonical identifiers, numeric values, protocol facets over split, metric surface, and unit,
publication years derived from arXiv identifier prefixes, and pointers consisting of an arXiv
identifier, an arXiv abstract URL, and a source location such as a table label.

### Is any information missing?

Yes, and the missingness is a first-class part of the measurement rather than a defect.
0.479 [0.399, 0.574] of entries visible at the final cutoff sit in no valid comparison set
because at least one must-agree protocol facet is not recoverable from the paper text. 26 of
99 families never reach a justified conclusion at any cutoff. Six inventory dimensions
(evaluation setting, training data, preprocessing, dataset version, aggregation scope, and
compute) are missing for every entry, which is the honest state of the corpus.

### Does it contain confidential or offensive content, or data about people?

No. Every instance is a numeric result reported in a public arXiv preprint. Author names are
not included. No personal data is present.

### Are there errors, noise, or redundancies?

Yes, and they are quantified rather than hidden.

- Every dispersion is a calibrated default, because the substrate carries no evidence text.
  The share of beyond-noise calls made on reported dispersion is zero.
- Upstream value extraction is ambiguous on multi-number table rows. A model reading the same
  span agrees with the recorded value on 0.763 [0.727, 0.796] of propagation pairs.
- The upstream `split` field is a pipeline inference rather than a reading of the span. Spans
  consisting of a bare number still carry `split = test`. Consumers should treat it as an
  assigned label, not a stated one.
- The same source number may be quoted by several citing papers, so propagation pairs are not
  independent. All propagation rates are bootstrapped over cells for this reason.

### Is it self-contained?

The resource is self-contained for analysis but deliberately not for verification. It carries
pointers, not text. Verifying any individual number requires consulting the cited arXiv paper.
This is required by the substrate licence.

## Collection

### How was the data acquired?

It is derived, not collected. The replay is computed deterministically from the frozen Study
3 cleaned-leaderboard resource, sha256
`b2dd904793a6f5dac0815a2936553d67aac2005e2dbb1cdc735fd79073ef2472`, itself built from a
Papers-with-Code archive snapshot frozen 2025-07-28. Publication years come from arXiv
identifier prefixes, with 100 percent coverage. A robustness join against the frozen
Papers-with-Code papers table agrees on 34,189 of 34,247 entries, that is 99.83 percent, and
leaves the qualifying family set identical.

Only the citation-propagation measurement used a language model, and only to read spans that
were already extracted upstream. No new corpus was fetched.

### Over what timeframe?

Entries span publication years 2014 to 2025. The replay was computed in July 2026.

### Were ethical review processes conducted?

No human subjects are involved, so no review was sought. The resource redistributes no text.

## Preprocessing

### What preprocessing was done?

Comparability partitioning is consumed frozen from the upstream resource and was not recomputed. The only
value normalisation applied is a deterministic percent-versus-fraction reconciliation. The
noise model is the upstream preregistered model, imported rather than copied so that this
resource cannot drift from the instrument it reuses.

### Is the raw data available?

The upstream substrate and extraction are the raw data and are separately released under the
same licence. Every derived quantity here is reproducible from them by the scripts in the
repository.

## Uses

### What is it intended for?

Studying the temporal stability of computed scientific conclusions; evaluating update policies
against retrospective regret; measuring protocol drift and unknown comparability at corpus
scale; and studying whether protocol qualifiers survive citation.

### What should it not be used for?

1. **Do not use it as a leaderboard.** The declared winner at a cutoff is a computed artefact
   under a stated policy, not a claim about which method is best.
2. **Do not read the unsettled-transition rate as a stable property.** It ranges from 0.471 to
   0.964 across a noise-band sweep. The band-invariant quantities, namely the zero reversal
   count and the protocol-attributable count, are the ones that support claims.
3. **Do not read `weakened` as a protocol phenomenon.** It is consistent with benchmark
   saturation and top-crowding.
4. **Do not read the propagation pairs as verified citation edges.** They are numeric
   co-occurrences of the same canonical cell across papers with a plausible citation relation.
   No reference list was parsed.
5. **Do not conclude that citing papers never state the evaluation split.** The unit of
   observation is the span carrying the number, not the whole paper.

### Are there groups this could unfairly affect?

The resource names methods and papers, and records where a quoted number differs from its
source. Any such difference is most often a legitimate configuration difference: restricted to
pairs an adjudicator judged to be the same quantity, numbers are copied correctly
0.895 [0.852, 0.933] of the time. Individual rows should not be used to allege misconduct by
any author. The measurement is about a systemic reporting practice, namely omission of the
protocol qualifier, and not about individual conduct.

## Distribution and maintenance

Distributed with the project repository under CC-BY-SA 4.0, matching the substrate.
`MANIFEST.json` pins a sha256 for every file. The resource is a frozen snapshot tagged
`v1.0-experiments` and is not intended to be updated in place; a changed substrate would be a
new version. Errata will be recorded in the repository `REPORT.md` under Deviations.

The dataset was assembled by the author. There is no separate maintenance team, and no support
commitment is implied.

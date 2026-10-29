# Manual adjudication of the Phase 3 extraction, 47 stratified pairs

**Status of this record.** The 47 items in `adjudication_sheet.json` were read and adjudicated
by the execution agent, not by an independent human coder. It is labelled that way throughout
and should be treated as an internal instrument check awaiting author confirmation, not as
human gold. The sheet is stratified by the adjudicator relation and the copy-fidelity class,
seed 13, up to 6 items per stratum.

The adjudication answers one question per item: **does the citing span, that is the excerpt
that actually carries the number, state the evaluation split?**

## Result 1: the model's split judgment agrees with adjudication on 47 of 47 items

Forty-six of the forty-seven citing spans state no evaluation split. The model recorded
`split_is_explicit: false` on all forty-six.

One span states a split. Item 500, on ActivityNet-1.3 AUC, reads `AUC (val) | 59.58 | ... `,
and the model recorded `stated_split: "val", split_is_explicit: true`. It was the only such
span in the sheet and the model caught it.

Agreement is therefore 47 of 47, including the single positive case. The measured rate of
0.019 [0.008, 0.032] for explicit split statement is consistent with what the spans contain.

## Result 2: the upstream `split` field is a pipeline inference, not a reading of the span

This is the consequential finding of the adjudication, and it changes how the deterministic
census must be read.

For the large majority of the forty-six spans with no stated split, the frozen the upstream resource
extraction nevertheless records `split = test`. Representative examples, quoted in full:

| item | citing span | the upstream resource split |
|---|---|---|
| 171 | `90.02` | test |
| 586 | `27.35` | test |
| 511 | `MMGCN  \|  0.710` | test |
| 626 | `ConvE [ 7 ] \| 5277 \| .46 \| .39 \| .43 \| .48` | test |
| 645 | `APPNP \| 79.78 ± 0.33 ...` | test |

A span reading `90.02` cannot support the reading `test`. The label must come from a table
caption, from surrounding section context, or from a default applied when no split is
observed. Nothing here is wrong with the upstream pipeline, whose split field was built for a
different purpose, but it means the deterministic qualifier-preservation census in section 8.1
measures the propagation of a **pipeline-assigned** split rather than of a **stated** one.

Consequence, applied in the report: the deterministic `validation_cited_as_test` rate of
0.400 [0.217, 0.571] is not reported as the headline, because its citing-side label is
inferred. The claim that survives adjudication is the model-measured one, namely that the span
quoting a validation-split result omits any evaluation qualifier on 0.959 [0.891, 1.000] of
such pairs.

## Result 3: value extraction is ambiguous on multi-number table rows

The model reads the same number as the upstream extraction on 0.763 [0.727, 0.796] of pairs.
Adjudication shows the disagreements are concentrated where a span is a table row carrying many
numbers and it is genuinely undecidable from the span alone which one belongs to the cell.

- Item 690, `GCN | 86.10 | 86.50 | 78.70 | 85.80 | 71.60 | 74.60`. the upstream resource took 85.80, the
  model took 86.10. Neither is demonstrably right from the span.
- Item 409, `DLinear | 0.381 | 0.405 | 0.398 | 0.418 | 0.375 | 0.399`. the upstream resource records 0.331,
  which does not appear in the span at all.
- Item 386, `byte mLSTM ... | 86.8 | 90.6 | 94.7 | 88.8 | 90.4 | 91.7 | 54.6 | 92.2`. the upstream resource
  took 94.7, the model took 92.2.

The model returns a null value rather than guessing on several ambiguous rows, for example
items 439, 593, and 428, which is the conservative behaviour the prompt asks for.

Consequence, applied in the report: the copy-fidelity numbers carry an extraction-ambiguity
floor and are reported as bounded rather than exact. No claim in the study rests on the raw
copy-fidelity mismatch rate.

## Result 4: the same-quantity adjudicator behaves as intended

The adjudicator separates the two populations cleanly. Copy mismatch is 0.105 [0.067, 0.148]
among pairs it calls same quantity and 0.952 [0.912, 0.981] among pairs it calls a different
configuration. A classifier that had no signal could not produce that separation.

The `cannot_tell` rate of 0.309 is honest rather than evasive. Adjudication of the
`cannot_tell` items confirms that the spans genuinely do not determine the configuration, for
example item 170, where the span gives a ScanObjectNN accuracy without saying which of the
OBJ-BG, OBJ-ONLY, or PB-T50-RS variants it refers to.

## Result 5: the split judgment is not backbone-specific

Repeating the extraction pass on `deepseek/deepseek-v4-flash`, an independent backbone, gives
agreement with `anthropic/claude-sonnet-5` on whether a span states the split of 0.996
[0.991, 1.000] over the 781 pairs both answered, Cohen kappa 0.895 against base rates of 0.018
and 0.019. Each model independently flags 15 explicit spans, and on the 13 both call explicit
they name the same split 13 of 13 times. Two models agreeing does not make them right, and
this is not a substitute for human validation, but it removes the reading that the near-zero
qualifier rate is an artefact of one model family.

## What this adjudication does not establish

It does not establish that the citing paper never states the split anywhere. It establishes
that the split is not stated in the span that carries the number. Those are different claims,
and only the second is made in the report.

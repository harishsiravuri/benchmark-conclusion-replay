"""Phase 3b: language-model reading of citing spans. The only phase that spends.

Two passes, deliberately separated so that neither biases the other.

  Pass A, extraction.  The model sees ONLY the citing evidence span and the canonical cell.
                       It reports the number the span attributes to that method, the
                       evaluation split as stated in the span, and any other stated
                       condition. The source reading is never shown, so the model cannot
                       anchor on what the source paper said. This pass is what measures
                       qualifier statement.
  Pass B, adjudication. The model sees both spans and decides whether the citing paper is
                       repeating the same quantity or reporting a different configuration of
                       the method. This separates citation distortion from a legitimately
                       different number, which the deterministic census cannot do.

Both passes are cached by sha256 of the full request, so a rerun costs nothing.
"""

from __future__ import annotations

import json

EXTRACT_SYSTEM = """\
You read a short excerpt from a scientific paper and report only what that excerpt states.

You will be given a canonical result cell, that is a method, a dataset, and a metric, and one
excerpt from a paper that mentions a number for that cell. The excerpt may be a table caption,
a table row, or a sentence.

Report, strictly from the excerpt and from nothing else:

1. the numeric value the excerpt attributes to that method on that dataset and metric;
2. the evaluation split the excerpt states for that number;
3. any other evaluation condition the excerpt states, for example an input resolution, a
   backbone, a number of shots, an amount of extra training data, or a variant name;
4. whether the excerpt explicitly states the evaluation split at all.

Rules. Do not infer an evaluation split from convention, from the dataset name, or from what
is usual in the field. If the excerpt does not say which split the number is on, then the
split is not stated, and you must record that. Report a value only if the excerpt actually
carries a number for that method on that cell. Never guess a number.

Answer with a single JSON object and no other text:

{"cited_value": <number or null>,
 "stated_split": "test" | "val" | "dev" | "train" | "other" | null,
 "split_is_explicit": true | false,
 "other_stated_conditions": "<short phrase, or null>",
 "value_found": true | false,
 "confidence": <number between 0 and 1>}
"""

ADJUDICATE_SYSTEM = """\
You compare two excerpts that both report a number for the same canonical result cell, that
is the same method, dataset, and metric.

The first excerpt is from the paper that originally reported the result. The second is from a
later paper that repeats the number, that is it cites the result rather than producing it.

Decide whether the later paper is reporting THE SAME QUANTITY as the original, or a DIFFERENT
CONFIGURATION of the same method, for example a different backbone, a different input
resolution, a different number of shots, a different amount of training data, a different
split, or a different variant of the method.

This distinction matters. If the two numbers differ because the later paper is quoting a
different configuration, that is not a citation error. If the two numbers differ while
referring to the same quantity, that is a copying discrepancy.

Answer with a single JSON object and no other text:

{"relation": "same_quantity" | "different_configuration" | "cannot_tell",
 "differing_condition": "<the condition that differs, or null>",
 "rationale": "<one short sentence>",
 "confidence": <number between 0 and 1>}
"""


def extract_user(pair) -> str:
    return json.dumps({
        "cell": {"method": pair.cell_names[0], "dataset": pair.cell_names[1],
                 "metric": pair.cell_names[2]},
        "excerpt_from_citing_paper": (pair.citing_quote or "")[:4000],
    }, ensure_ascii=False, indent=2)


def adjudicate_user(pair) -> str:
    return json.dumps({
        "cell": {"method": pair.cell_names[0], "dataset": pair.cell_names[1],
                 "metric": pair.cell_names[2]},
        "original_paper": {
            "excerpt": (pair.source_quote or "")[:3000],
            "reported_value": pair.source_value,
        },
        "later_paper": {
            "excerpt": (pair.citing_quote or "")[:3000],
            "reported_value": pair.citing_value,
        },
    }, ensure_ascii=False, indent=2)


def estimate_tokens(text: str) -> int:
    """Deliberately generous character-based estimate, used only for the cost projection."""
    return max(1, len(text) // 3)


def project_cost(pairs, system: str, user_fn, price_in_per_m: float,
                 price_out_per_m: float, expected_out_tokens: int = 220) -> dict:
    tin = sum(estimate_tokens(system) + estimate_tokens(user_fn(p)) for p in pairs)
    tout = expected_out_tokens * len(pairs)
    cost = tin / 1e6 * price_in_per_m + tout / 1e6 * price_out_per_m
    return {"n_calls": len(pairs), "est_tokens_in": tin, "est_tokens_out": tout,
            "projected_usd": cost}

"""Replay certificate: the the upstream resource certificate schema extended for chronological replay.

One certificate is emitted per family, per cutoff, per policy. It reuses the the upstream resource
certificate fields (query, cell, per-facet relations, compatible and incompatible and missing
dimensions, status, provenance-only flag) and adds the four fields the study protocol
requires: the cutoff year, the policy, the evidence set size, and the transition status.

Provenance is pointers only. No excerpt text is carried, which matches the CC-BY-SA 4.0 terms
of the substrate release.
"""

from __future__ import annotations

CERTIFICATE_VERSION = "replay-1.0"

# The must-agree facet set, that is the Q-official style constraint the the upstream resource clustering
# already enforces.
MUST_AGREE_FACETS = ["split", "metric_surface", "unit"]

# Inventory dimensions the extraction pipeline cannot recover at scale (the upstream resource framework,
# section 4). They are missing for every entry in this substrate, and saying so is the honest
# statement the study has to make.
NEVER_OBSERVED_DIMENSIONS = [
    "evaluation_setting", "training_data", "preprocessing",
    "dataset_version", "aggregation_scope", "compute",
]


def build_replay_certificate(family, conclusion, evidence, transition=None) -> dict:
    """One machine-readable certificate for a (family, cutoff, policy) triple."""
    protocol = conclusion.principal_signature or {}
    observed = [f for f in MUST_AGREE_FACETS if protocol.get(f) is not None]
    missing = [f for f in MUST_AGREE_FACETS if protocol.get(f) is None]

    winner_entry = None
    if conclusion.winner_method is not None:
        pool = (evidence.clusters.get(conclusion.winner_cluster, ())
                if conclusion.winner_cluster is not None else evidence.entries)
        for e in pool:
            if e.method == conclusion.winner_method and e.value == conclusion.winner_value:
                winner_entry = e
                break

    return {
        "certificate_version": CERTIFICATE_VERSION,
        "query": {
            "template": "Q_official_style" if conclusion.policy != "naive_latest_max" else "Q_any",
            "must_agree_facets": MUST_AGREE_FACETS if conclusion.policy != "naive_latest_max" else [],
        },
        "cell": {"dataset": family.dataset, "metric": family.metric,
                 "metric_direction": family.metric_direction,
                 "leaderboard_id": family.leaderboard_id},
        # --- replay extension --------------------------------------------------------
        "cutoff_year": conclusion.cutoff,
        "policy": conclusion.policy,
        "evidence_set_size": conclusion.evidence_size,
        "transition_status": transition.status if transition is not None else None,
        "transition_reason": transition.reason if transition is not None else None,
        # -----------------------------------------------------------------------------
        "conclusion": {
            "winner_method": conclusion.winner_method,
            "winner_value": conclusion.winner_value,
            "winner_cluster": conclusion.winner_cluster,
            "answer_status": conclusion.answer_status,
            "margin_over_runner_up": conclusion.margin_over_runner_up,
            "margin_beyond_noise": conclusion.margin_beyond_noise,
        },
        "winner_provenance": (
            {"arxiv_id": winner_entry.arxiv_id,
             "arxiv_abs_url": f"https://arxiv.org/abs/{winner_entry.arxiv_id}",
             "source_location": winner_entry.source_location}
            if winner_entry is not None else None
        ),
        "comparison_set": {
            "size": conclusion.comparison_set_size,
            "n_valid_sets": conclusion.n_valid_sets,
            "unknown_size": conclusion.unknown_size,
            "unknown_share": conclusion.unknown_share,
            "shared_methods_across_sets": conclusion.shared_methods_across_sets,
            "protocol_signature": protocol or None,
        },
        "compatible_dimensions": observed,
        "incompatible_dimensions": [],
        "missing_dimensions": missing + NEVER_OBSERVED_DIMENSIONS,
        "cluster_winners": conclusion.cluster_winners,
        "notes": list(conclusion.notes),
        "provenance_only": True,
        "evidence_text_redistributed": False,
    }


def self_consistent(cert: dict) -> bool:
    """A certificate must be reconstructable from the fields it carries.

    A withheld conclusion has exactly two admissible causes, and both must be visible in the
    certificate itself: either no valid comparison set exists at the cutoff, or the metric
    direction is unknown so that an argmax is undefined. The second case is recorded by the
    answer status `undefined`.
    """
    cs = cert["comparison_set"]
    concl = cert["conclusion"]
    if concl["winner_method"] is None:
        if concl["answer_status"] == "undefined":
            return cert["cell"]["metric_direction"] not in ("higher", "lower")
        return cs["size"] == 0 or cert["policy"] != "protocol_partitioned"
    if cert["policy"] == "naive_latest_max":
        return cs["size"] == cert["evidence_set_size"]
    return 0 < cs["size"] <= cert["evidence_set_size"]

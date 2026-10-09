"""Adapter onto the upstream preregistered noise model, imported read-only.

The the upstream resource model lives at `upstream/src/noise/model.py` and is imported rather than
copied, so this study cannot drift from the instrument it claims to reuse. Its sha256 is
recorded in every checkpoint.

One honest limitation is recorded here and reported in every downstream artifact. The
cleaned-leaderboard substrate is pointer only and carries no evidence text, so no reported
dispersion string of the form "X plus or minus S" is available to parse. Every dispersion
used by this study is therefore the upstream calibrated DEFAULT. Under that default the
beyond-noise threshold is 2.0 normalized points for bounded metrics and about 5.94 percent
relative for unbounded metrics.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache

from replaykg.paths import CH3_NOISE_MODEL, PAPER32_SRC, sha256_file


@lru_cache(maxsize=1)
def _load_ch3_noise():
    if str(PAPER32_SRC) not in sys.path:
        sys.path.insert(0, str(PAPER32_SRC))
    spec = importlib.util.spec_from_file_location("ch3_noise_model", CH3_NOISE_MODEL)
    module = importlib.util.module_from_spec(spec)
    # registered before execution because `dataclasses` resolves a class's module by name
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def noise_model_provenance() -> dict:
    m = _load_ch3_noise()
    return {
        "source": str(CH3_NOISE_MODEL),
        "sha256": sha256_file(CH3_NOISE_MODEL),
        "k": m.K,
        "sigma_default_rel": m.SIGMA_DEFAULT_REL,
        "sigma_default_bounded_pts": m.SIGMA_DEFAULT_BOUNDED_PTS,
        "dispersion_source": "defaulted for every call (substrate is pointer only)",
    }


def classify_metric_range(metric_name: str) -> str:
    return _load_ch3_noise().classify_metric_range(metric_name)


def beyond_noise(v1: float, v2: float, metric_name: str, band_scale: float = 1.0) -> bool:
    """True when the two values differ by more than the the upstream resource noise band.

    `band_scale` multiplies the band and exists only for the preregistered sensitivity
    analysis. The primary analysis always uses 1.0, that is the the upstream resource band unmodified.
    """
    if v1 is None or v2 is None:
        return False
    d = _load_ch3_noise().decide(v1, v2, metric_name)
    if band_scale == 1.0:
        return bool(d.beyond_noise)
    if d.decision_units == "both(AND)":
        # the unknown-range rule requires both views; scaling one threshold is ill defined,
        # so the conservative choice is to keep the unmodified decision
        return bool(d.beyond_noise)
    return bool(d.gap > d.threshold * band_scale)


def noise_detail(v1: float, v2: float, metric_name: str) -> dict:
    d = _load_ch3_noise().decide(v1, v2, metric_name)
    return {
        "beyond_noise": bool(d.beyond_noise),
        "range_type": d.range_type,
        "threshold": d.threshold,
        "gap": d.gap,
        "decision_units": d.decision_units,
        "sd_source": f"{d.sd_source_left}/{d.sd_source_right}",
    }

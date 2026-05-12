# morie.fn -- function file (hadesllm/morie)
"""SDM pseudo-R-squared."""

import numpy as np

from ._containers import SpatialResult


def sdmr2(ll_model, ll_null, n):
    """SDM pseudo-R-squared.

    Category: SDM

    Parameters
    ----------
    ll_model, ll_null, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        cox_snell = 1.0 - np.exp(2.0 / n * (ll_null - ll_model))
        max_cs = 1.0 - np.exp(2.0 / n * ll_null)
        result = float(cox_snell / max_cs) if abs(max_cs) > 1e-12 else 0.0
        return SpatialResult(name="sdmr2", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sdmr2", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sdmr2_fn = sdmr2


def cheatsheet() -> str:
    return "sdmr2({}) -> SDM pseudo-R-squared."

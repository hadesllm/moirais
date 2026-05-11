# morie.fn — function file (hadesllm/morie)
"""SAR pseudo-R-squared (Nagelkerke)."""

import numpy as np

from ._containers import SpatialResult


def sarr2(ll_model, ll_null, n):
    """SAR pseudo-R-squared (Nagelkerke).

    Category: SAR

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
        return SpatialResult(name="sarr2", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sarr2", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sarr2_fn = sarr2


def cheatsheet() -> str:
    return "sarr2({}) -> SAR pseudo-R-squared (Nagelkerke)."

# morie.fn — function file (hadesllm/morie)
"""Gravity model LM test for spatial autocorrelation."""

import numpy as np

from ._containers import SpatialResult


def igravlm(resid, W):
    """Gravity model LM test for spatial autocorrelation.

    Category: Gravity

    Parameters
    ----------
    resid, W : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(resid)
        Wresid = np.dot(W, resid)
        result = float(np.dot(resid, Wresid) / (np.dot(resid, resid) + 1e-12))
        return SpatialResult(name="igravlm", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="igravlm", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


igravlm_fn = igravlm


def cheatsheet() -> str:
    return "igravlm({}) -> Gravity model LM test for spatial autocorrelation."

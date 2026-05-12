# morie.fn -- function file (hadesllm/morie)
"""SAR score / LM test for spatial lag."""

import numpy as np

from ._containers import SpatialResult


def sarsc(resid, W):
    """SAR score / LM test for spatial lag.

    Category: SAR

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
        return SpatialResult(name="sarsc", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sarsc", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sarsc_fn = sarsc


def cheatsheet() -> str:
    return "sarsc({}) -> SAR score / LM test for spatial lag."

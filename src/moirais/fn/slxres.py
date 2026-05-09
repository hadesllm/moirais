"""SLX residual Moran test."""

import numpy as np

from ._containers import SpatialResult


def slxres(resid, W):
    """SLX residual Moran test.

    Category: SLX

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
        return SpatialResult(name="slxres", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="slxres", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


slxres_fn = slxres


def cheatsheet() -> str:
    return "slxres({}) -> SLX residual Moran test."

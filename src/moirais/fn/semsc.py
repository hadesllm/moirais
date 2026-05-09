# moirais.fn — function file (hadesllm/moirais)
"""SEM score test for spatial error parameter."""

import numpy as np

from ._containers import SpatialResult


def semsc(resid, W):
    """SEM score test for spatial error parameter.

    Category: SEM

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
        return SpatialResult(name="semsc", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="semsc", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


semsc_fn = semsc


def cheatsheet() -> str:
    return "semsc({}) -> SEM score test for spatial error parameter."

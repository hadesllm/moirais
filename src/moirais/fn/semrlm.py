# moirais.fn — function file (hadesllm/moirais)
"""SEM robust LM test for spatial error."""

import numpy as np

from ._containers import SpatialResult


def semrlm(resid, W, X):
    """SEM robust LM test for spatial error.

    Category: SEM

    Parameters
    ----------
    resid, W, X : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(resid)
        Wresid = np.dot(W, resid)
        result = float(np.dot(resid, Wresid) / (np.dot(resid, resid) + 1e-12))
        return SpatialResult(name="semrlm", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="semrlm", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


semrlm_fn = semrlm


def cheatsheet() -> str:
    return "semrlm({}) -> SEM robust LM test for spatial error."

# morie.fn -- function file (hadesllm/morie)
"""SEM LM test for spatial error."""

import numpy as np

from ._containers import SpatialResult


def semlm(resid, W):
    """SEM LM test for spatial error.

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
        return SpatialResult(name="semlm", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="semlm", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


semlm_fn = semlm


def cheatsheet() -> str:
    return "semlm({}) -> SEM LM test for spatial error."

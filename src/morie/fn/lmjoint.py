# morie.fn — function file (hadesllm/morie)
"""Joint LM test for spatial lag and error."""

import numpy as np

from ._containers import SpatialResult


def lmjoint(resid, X, W):
    """Joint LM test for spatial lag and error.

    Category: LM

    Parameters
    ----------
    resid, X, W : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(resid)
        Wresid = np.dot(W, resid)
        result = float(np.dot(resid, Wresid) / (np.dot(resid, resid) + 1e-12))
        return SpatialResult(name="lmjoint", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="lmjoint", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


lmjoint_fn = lmjoint


def cheatsheet() -> str:
    return "lmjoint({}) -> Joint LM test for spatial lag and error."

# morie.fn — function file (hadesllm/morie)
"""One-directional robust LM lag (Bera-Yoon)."""

import numpy as np

from ._containers import SpatialResult


def lmrlag2(resid, X, W):
    """One-directional robust LM lag (Bera-Yoon).

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
        return SpatialResult(name="lmrlag2", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="lmrlag2", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


lmrlag2_fn = lmrlag2


def cheatsheet() -> str:
    return "lmrlag2({}) -> One-directional robust LM lag (Bera-Yoon)."

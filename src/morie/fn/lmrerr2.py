# morie.fn -- function file (hadesllm/morie)
"""One-directional robust LM error (Bera-Yoon)."""

import numpy as np

from ._containers import SpatialResult


def lmrerr2(resid, X, W):
    """One-directional robust LM error (Bera-Yoon).

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
        return SpatialResult(name="lmrerr2", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="lmrerr2", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


lmrerr2_fn = lmrerr2


def cheatsheet() -> str:
    return "lmrerr2({}) -> One-directional robust LM error (Bera-Yoon)."

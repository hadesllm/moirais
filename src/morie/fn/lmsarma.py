# morie.fn — function file (hadesllm/morie)
"""LM test for SARMA (lag + error)."""

import numpy as np

from ._containers import SpatialResult


def lmsarma(resid, X, W):
    """LM test for SARMA (lag + error).

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
        return SpatialResult(name="lmsarma", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="lmsarma", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


lmsarma_fn = lmsarma


def cheatsheet() -> str:
    return "lmsarma({}) -> LM test for SARMA (lag + error)."

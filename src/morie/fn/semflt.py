# morie.fn -- function file (hadesllm/morie)
"""SEM Cochrane-Orcutt spatial filter transform."""

import numpy as np

from ._containers import SpatialResult


def semflt(y, W, lam=0.3):
    """SEM Cochrane-Orcutt spatial filter transform.

    Category: SEM

    Parameters
    ----------
    y, W, lam=0.3 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        Wy = np.dot(W, y)
        result = float(np.corrcoef(y, Wy)[0, 1])
        return SpatialResult(name="semflt", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="semflt", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


semflt_fn = semflt


def cheatsheet() -> str:
    return "semflt({}) -> SEM Cochrane-Orcutt spatial filter transform."

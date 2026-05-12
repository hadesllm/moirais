# morie.fn -- function file (hadesllm/morie)
"""SDM common-factor restriction test (Wald)."""

import numpy as np

from ._containers import SpatialResult


def sdmcf(coef, theta, vcov):
    """SDM common-factor restriction test (Wald).

    Category: SDM

    Parameters
    ----------
    coef, theta, vcov : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        diff = coef - theta if len(coef) == len(theta) else coef
        k = len(diff)
        V = vcov[:k, :k] if vcov.shape[0] > k else vcov
        result = float(diff @ np.linalg.solve(V + 1e-8 * np.eye(k), diff))
        return SpatialResult(name="sdmcf", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sdmcf", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sdmcf_fn = sdmcf


def cheatsheet() -> str:
    return "sdmcf({}) -> SDM common-factor restriction test (Wald)."

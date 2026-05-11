# morie.fn — function file (hadesllm/morie)
"""SDEM common-factor restriction test."""

import numpy as np

from ._containers import SpatialResult


def sdemcf(coef, theta, vcov):
    """SDEM common-factor restriction test.

    Category: SDEM

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
        return SpatialResult(name="sdemcf", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="sdemcf", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


sdemcf_fn = sdemcf


def cheatsheet() -> str:
    return "sdemcf({}) -> SDEM common-factor restriction test."

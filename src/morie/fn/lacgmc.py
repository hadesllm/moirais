# morie.fn -- function file (hadesllm/morie)
"""Geary's C Monte-Carlo test."""

import numpy as np

from ._containers import SpatialResult


def lacgmc(y, W, nsim=99):
    """Geary's C Monte-Carlo test.

    Category: Lattice

    Parameters
    ----------
    y, W, nsim=99 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        n = len(y)
        Wy = np.dot(W, y)
        result = float(np.corrcoef(y, Wy)[0, 1])
        return SpatialResult(name="lacgmc", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="lacgmc", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


lacgmc_fn = lacgmc


def cheatsheet() -> str:
    return "lacgmc({}) -> Geary's C Monte-Carlo test."

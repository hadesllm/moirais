# morie.fn -- function file (hadesllm/morie)
"""Join count Monte-Carlo test."""

import numpy as np

from ._containers import SpatialResult


def lacjmc(y_binary, W, nsim=99):
    """Join count Monte-Carlo test.

    Category: Lattice

    Parameters
    ----------
    y_binary, W, nsim=99 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        y_arr = np.asarray(y_binary, dtype=float)
        sims = [
            float(np.dot(np.random.permutation(y_arr), np.dot(W, np.random.permutation(y_arr)))) for _ in range(nsim)
        ]
        obs = float(np.dot(y_arr, np.dot(W, y_arr)))
        result = float(np.mean(np.array(sims) >= obs))
        return SpatialResult(name="lacjmc", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="lacjmc", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


lacjmc_fn = lacjmc


def cheatsheet() -> str:
    return "lacjmc({}) -> Join count Monte-Carlo test."

# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian credible interval (equal-tailed or HDI)."""

from __future__ import annotations

from typing import Union

import numpy as np


def bayesian_credible_interval(
    samples: Union[list, np.ndarray],
    *,
    alpha: float = 0.05,
    method: str = "equal_tailed",
) -> tuple[float, float]:
    """
    Compute a Bayesian credible interval from posterior samples.

    Two methods:

    * ``"equal_tailed"`` -- quantile-based: (alpha/2, 1 - alpha/2)
      percentiles.
    * ``"hdi"`` -- Highest Density Interval (shortest interval containing
      1 - alpha probability mass).

    :param samples: 1-D array of posterior samples.
    :param alpha: Significance level (default 0.05 for 95% interval).
    :param method: ``"equal_tailed"`` or ``"hdi"``.
    :return: Tuple (lower, upper).
    :raises ValueError: If method unknown, alpha not in (0, 1), or too few
        samples.

    References
    ----------
    Kruschke, J. K. (2015). *Doing Bayesian Data Analysis* (2nd ed.).
    Academic Press.
    """
    arr = np.asarray(samples, dtype=float)
    if len(arr) < 3:
        raise ValueError("Need at least 3 samples.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1).")

    if method == "equal_tailed":
        lo = float(np.percentile(arr, 100 * alpha / 2))
        hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
        return (lo, hi)
    elif method == "hdi":
        from morie.fn.hdi import highest_density_interval

        return highest_density_interval(arr, mass=1.0 - alpha)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'equal_tailed' or 'hdi'.")


bci = bayesian_credible_interval


def cheatsheet() -> str:
    return "bayesian_credible_interval({}) -> Bayesian credible interval (equal-tailed or HDI)."

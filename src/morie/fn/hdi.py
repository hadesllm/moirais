# morie.fn -- function file (hadesllm/morie)
"""Highest Density Interval from MCMC samples."""

from __future__ import annotations

from typing import Union

import numpy as np


def highest_density_interval(
    samples: Union[list, np.ndarray],
    *,
    mass: float = 0.95,
) -> tuple[float, float]:
    """
    Compute the Highest Density Interval (HDI) from posterior samples.

    The HDI is the shortest interval containing the specified probability
    mass. For unimodal posteriors this is equivalent to finding the interval
    [a, b] such that P(a <= theta <= b) = mass and (b - a) is minimised.

    Algorithm: sort the samples, compute all contiguous intervals of length
    ceil(mass * n), and return the one with smallest width.

    :param samples: 1-D array of posterior samples.
    :param mass: Probability mass to include (default 0.95).
    :return: Tuple (lower, upper) of the HDI bounds.
    :raises ValueError: If mass not in (0, 1) or fewer than 3 samples.

    References
    ----------
    Kruschke, J. K. (2015). *Doing Bayesian Data Analysis* (2nd ed.).
    Academic Press.

    Chen, M.-H., & Shao, Q.-M. (1999). Monte Carlo estimation of Bayesian
    credible and HPD intervals. *Journal of Computational and Graphical
    Statistics*, 8(1), 69--92.
    """
    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 1 or len(arr) < 3:
        raise ValueError("samples must be a 1-D array with at least 3 elements.")
    if not (0 < mass < 1):
        raise ValueError("mass must be in (0, 1).")

    sorted_pts = np.sort(arr)
    n = len(sorted_pts)
    n_included = int(np.ceil(mass * n))
    n_intervals = n - n_included

    widths = sorted_pts[n_included:] - sorted_pts[:n_intervals]
    min_idx = int(np.argmin(widths))

    return (float(sorted_pts[min_idx]), float(sorted_pts[min_idx + n_included]))


hdi = highest_density_interval


def cheatsheet() -> str:
    return "highest_density_interval({}) -> Highest Density Interval from MCMC samples."

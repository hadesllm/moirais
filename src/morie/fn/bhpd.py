# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Highest Posterior Density (HPD) interval."""

from __future__ import annotations

__all__ = ["hpd_interval", "bhpd"]

from typing import Any, Union

import numpy as np


def hpd_interval(
    samples: Union[list, np.ndarray],
    *,
    prob: float = 0.95,
) -> dict[str, Any]:
    """
    Highest Posterior Density (HPD) interval from posterior samples.

    The HPD interval is the shortest interval containing the
    specified probability mass.  For unimodal posteriors this is
    unique; for multimodal posteriors this returns the shortest
    contiguous interval.

    Algorithm: sort samples, compute all intervals of width
    floor(prob * n), return the narrowest.

    Parameters
    ----------
    samples : array-like
        Posterior samples (1-D).
    prob : float
        Probability mass (default 0.95).

    Returns
    -------
    dict
        hpd_lower, hpd_upper, width, prob, mean, median

    Raises
    ------
    ValueError
        If prob not in (0, 1) or samples too short.

    References
    ----------
    Chen, M.-H. & Shao, Q.-M. (1999). Monte Carlo estimation of
    Bayesian credible and HPD intervals. *Journal of Computational
    and Graphical Statistics*, 8(1), 69--92.
    Box, G. E. P. & Tiao, G. C. (1973). *Bayesian Inference in
    Statistical Analysis*, Wiley.
    """
    x = np.sort(np.asarray(samples, dtype=float).ravel())
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 samples.")
    if not 0 < prob < 1:
        raise ValueError("prob must be in (0, 1).")

    interval_size = max(int(np.floor(prob * n)), 1)
    if interval_size >= n:
        return {
            "hpd_lower": float(x[0]),
            "hpd_upper": float(x[-1]),
            "width": float(x[-1] - x[0]),
            "prob": prob,
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
        }

    widths = x[interval_size:] - x[: n - interval_size]
    best = int(np.argmin(widths))

    return {
        "hpd_lower": float(x[best]),
        "hpd_upper": float(x[best + interval_size]),
        "width": float(widths[best]),
        "prob": prob,
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
    }


bhpd = hpd_interval


def cheatsheet() -> str:
    return "hpd_interval(samples) -> Highest Posterior Density interval."

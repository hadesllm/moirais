# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian credible interval (equal-tailed)."""

from __future__ import annotations

__all__ = ["credible_interval", "bcred"]

from typing import Any, Union

import numpy as np


def credible_interval(
    samples: Union[list, np.ndarray],
    *,
    prob: float = 0.95,
) -> dict[str, Any]:
    """
    Equal-tailed Bayesian credible interval from posterior samples.

    The equal-tailed interval places (1-prob)/2 probability mass
    in each tail.

    Parameters
    ----------
    samples : array-like
        Posterior samples (1-D).
    prob : float
        Credible interval probability (default 0.95).

    Returns
    -------
    dict
        ci_lower, ci_upper, prob, median, mean, sd

    Raises
    ------
    ValueError
        If prob not in (0, 1) or samples empty.

    References
    ----------
    Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd ed.,
    CRC Press, Ch. 2.
    """
    x = np.asarray(samples, dtype=float).ravel()
    if len(x) == 0:
        raise ValueError("samples must not be empty.")
    if not 0 < prob < 1:
        raise ValueError("prob must be in (0, 1).")

    alpha = 1.0 - prob
    ci_lo = float(np.percentile(x, 100 * alpha / 2))
    ci_hi = float(np.percentile(x, 100 * (1 - alpha / 2)))

    return {
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "prob": prob,
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)),
    }


bcred = credible_interval


def cheatsheet() -> str:
    return "credible_interval(samples) -> Equal-tailed Bayesian credible interval."

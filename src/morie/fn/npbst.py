# morie.fn — function file (hadesllm/morie)
"""Nonparametric bootstrap inference for arbitrary statistics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def npbst(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    n_boot: int = 1999,
    alpha: float = 0.05,
    method: str = "percentile",
    seed: int = 42,
) -> dict:
    r"""
    Nonparametric bootstrap inference for an arbitrary statistic.

    Resamples rows of *data* with replacement *n_boot* times, computes
    *statistic* on each resample, and returns the point estimate,
    standard error, and confidence interval.

    Three interval methods are available:

    * ``"percentile"`` -- quantiles of the bootstrap distribution
      (Efron & Tibshirani, 1993, Ch. 13).
    * ``"basic"``      -- the *basic* (or *reverse percentile*) interval
      :math:`(2\\hat{\\theta} - q_{1-\\alpha/2},\\; 2\\hat{\\theta} - q_{\\alpha/2})`.
    * ``"normal"``     -- normal approximation
      :math:`\\hat{\\theta} \\pm z_{1-\\alpha/2} \\, \\widehat{\\mathrm{se}}_{\\mathrm{boot}}`.

    .. math::

        \\widehat{\\mathrm{se}}_{\\mathrm{boot}}
        = \\left[\\frac{1}{B-1}
          \\sum_{b=1}^{B}(\\hat{\\theta}_b^* - \\bar{\\theta}^*)^2
          \\right]^{1/2}

    :param data: Array of shape ``(n,)`` or ``(n, p)``.
    :param statistic: Callable that takes an array and returns a scalar.
    :param n_boot: Number of bootstrap replicates. Default 1999.
    :param alpha: Significance level. Default 0.05.
    :param method: ``"percentile"``, ``"basic"``, or ``"normal"``.
    :param seed: Random seed. Default 42.
    :return: dict with ``estimate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``n_boot``, ``method``.
    :raises ValueError: If *method* is unknown or *alpha* out of range.

    References
    ----------
    Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the
        Bootstrap*. Chapman & Hall/CRC.
    Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods
        in Econometrics*. Springer, Ch. 3.
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    valid = {"percentile", "basic", "normal"}
    if method not in valid:
        raise ValueError(f"method must be one of {valid}, got '{method}'.")

    data = np.asarray(data, dtype=float)
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    theta_hat = float(statistic(data))

    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_stats[b] = statistic(data[idx])

    se = float(np.std(boot_stats, ddof=1))
    lo_q = alpha / 2
    hi_q = 1 - alpha / 2

    if method == "percentile":
        ci_lo = float(np.percentile(boot_stats, lo_q * 100))
        ci_hi = float(np.percentile(boot_stats, hi_q * 100))
    elif method == "basic":
        ci_lo = 2 * theta_hat - float(np.percentile(boot_stats, hi_q * 100))
        ci_hi = 2 * theta_hat - float(np.percentile(boot_stats, lo_q * 100))
    else:
        from scipy.stats import norm

        z = norm.ppf(hi_q)
        ci_lo = theta_hat - z * se
        ci_hi = theta_hat + z * se

    return {
        "estimate": theta_hat,
        "se": se,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "n_boot": n_boot,
        "method": method,
    }


npbst_fn = npbst


def cheatsheet() -> str:
    return "npbst(data, statistic) -> Nonparametric bootstrap inference."

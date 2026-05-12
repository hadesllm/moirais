# morie.fn -- function file (hadesllm/morie)
"""Empirical likelihood confidence interval."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

__all__ = ["elci"]


def elci(
    x: np.ndarray,
    *,
    alpha: float = 0.05,
    grid_points: int = 200,
) -> dict[str, Any]:
    r"""
    Compute an empirical likelihood confidence interval for the mean.

    The EL CI is the set of :math:`\mu` values where the empirical
    likelihood ratio does not exceed the :math:`\chi^2_1` critical value:

    .. math::

        \{\mu : -2\log R(\mu) \le \chi^2_{1,\alpha}\}

    :param x: 1-D array of observations.
    :param alpha: Significance level. Default 0.05.
    :param grid_points: Grid size for CI search. Default 200.
    :return: Dict with ``ci_lower``, ``ci_upper``, ``mean``, ``grid``,
        ``log_ratios``, ``n``.
    :raises ValueError: If x is empty.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 14. Springer.
    Owen, A.B. (2001). *Empirical Likelihood*. Chapman & Hall/CRC.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("x must be non-empty.")

    from morie.fn.empll import empll

    n = x.size
    x_bar = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n))
    chi2_crit = stats.chi2.ppf(1.0 - alpha, df=1)

    grid = np.linspace(x_bar - 4 * se, x_bar + 4 * se, grid_points)
    log_ratios = np.zeros(grid_points)

    for k, mu in enumerate(grid):
        try:
            res = empll(x, mu0=mu, alpha=alpha)
            log_ratios[k] = res["log_ratio"]
        except Exception:
            log_ratios[k] = np.inf

    in_ci = log_ratios <= chi2_crit
    if np.any(in_ci):
        ci_lower = float(grid[in_ci][0])
        ci_upper = float(grid[in_ci][-1])
    else:
        ci_lower = x_bar - 1.96 * se
        ci_upper = x_bar + 1.96 * se

    return {
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "mean": x_bar,
        "grid": grid,
        "log_ratios": log_ratios,
        "n": n,
    }


def cheatsheet() -> str:
    return "elci({x}) -> Empirical likelihood confidence interval."

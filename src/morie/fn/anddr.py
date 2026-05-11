# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Anderson-Darling goodness-of-fit statistic."""

from __future__ import annotations

import numpy as np
from scipy import stats


def anderson_darling(
    x: np.ndarray,
    *,
    dist: str = "norm",
) -> dict:
    r"""
    Anderson-Darling goodness-of-fit test.

    The Anderson-Darling statistic weights tail discrepancies more
    heavily than Cramer-von Mises:

    .. math::

        A^2 = -n - \frac{1}{n}\sum_{i=1}^n (2i - 1)
        \left[\ln F_0(x_{(i)}) + \ln(1 - F_0(x_{(n+1-i)}))\right]

    :param x: 1-D array of observations (n >= 3).
    :param dist: Distribution to test against. One of ``"norm"``,
        ``"expon"``, ``"logistic"``, ``"gumbel"``, ``"gumbel_l"``,
        ``"gumbel_r"``. Default ``"norm"``.
    :return: dict with ``A2`` (statistic), ``critical_values`` (dict
        mapping significance level strings to values),
        ``significance_level`` (most stringent rejected level or None),
        ``n``.
    :raises ValueError: If x has fewer than 3 observations or dist unknown.

    References
    ----------
    Anderson, T.W. & Darling, D.A. (1952). Asymptotic theory of
        certain 'goodness-of-fit' criteria based on stochastic processes.
        *Ann. Math. Statist.*, 23(2), 193--212.
    Kosorok, M.R. (2008). *Introduction to Empirical Processes and
        Semiparametric Inference*, Ch. 2. Springer.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 3:
        raise ValueError("Anderson-Darling requires at least 3 observations.")

    valid_dists = {"norm", "expon", "logistic", "gumbel", "gumbel_l", "gumbel_r"}
    if dist not in valid_dists:
        raise ValueError(f"dist must be one of {valid_dists}, got '{dist}'.")

    result = stats.anderson(x, dist=dist)

    sig_levels = [str(s) for s in result.significance_level]
    crit_vals = {s: float(c) for s, c in zip(sig_levels, result.critical_values)}

    rejected = None
    for s, c in zip(result.significance_level, result.critical_values):
        if result.statistic > c:
            rejected = str(s)

    return {
        "A2": float(result.statistic),
        "critical_values": crit_vals,
        "significance_level": rejected,
        "n": x.size,
        "dist": dist,
        "method": "Anderson-Darling test",
    }


anddr = anderson_darling


def cheatsheet() -> str:
    return "anderson_darling({x}) -> Anderson-Darling goodness-of-fit test."

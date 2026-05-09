# moirais.fn — function file (hadesllm/moirais)
"""Cramer-von Mises goodness-of-fit statistic."""

from __future__ import annotations

import numpy as np
from scipy import stats


def cramer_von_mises(
    x: np.ndarray,
    *,
    cdf: str = "norm",
    cdf_params: tuple = (),
) -> dict:
    r"""
    Cramer-von Mises goodness-of-fit test.

    The Cramer-von Mises statistic measures the integrated squared
    difference between the empirical and hypothesized CDFs:

    .. math::

        W^2 = \int_{-\infty}^{\infty}
        \left[\hat{F}_n(x) - F_0(x)\right]^2 \, dF_0(x)
        = \sum_{i=1}^n \left[\frac{2i - 1}{2n} - F_0(x_{(i)})\right]^2
        + \frac{1}{12n}

    :param x: 1-D array of observations.
    :param cdf: Null distribution name (scipy.stats). Default ``"norm"``.
    :param cdf_params: Extra parameters for the scipy distribution.
    :return: dict with ``W2`` (statistic), ``p_value``, ``reject``
        (at alpha=0.05), ``n``.
    :raises ValueError: If x has fewer than 3 observations or cdf unknown.

    References
    ----------
    Cramer, H. (1928). On the composition of elementary errors.
        *Skand. Aktuarietidskr.*, 11, 141--180.
    Kosorok, M.R. (2008). *Introduction to Empirical Processes and
        Semiparametric Inference*, Ch. 2. Springer.
    Anderson, T.W. (1962). On the distribution of the two-sample
        Cramer-von Mises criterion. *Ann. Math. Statist.*, 33, 1148--1159.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 3:
        raise ValueError("Cramer-von Mises requires at least 3 observations.")

    try:
        dist = getattr(stats, cdf)
    except AttributeError:
        raise ValueError(f"Unknown distribution: {cdf}.")

    result = stats.cramervonmises(x, cdf, args=cdf_params)

    return {
        "W2": float(result.statistic),
        "p_value": float(result.pvalue),
        "reject": float(result.pvalue) < 0.05,
        "n": x.size,
        "cdf": cdf,
        "method": "Cramer-von Mises test",
    }


cvmsv = cramer_von_mises


def cheatsheet() -> str:
    return "cramer_von_mises({x}) -> Cramer-von Mises goodness-of-fit test."

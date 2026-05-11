# morie.fn — function file (hadesllm/morie)
"""Cauchy distribution cumulative distribution function."""

from typing import Union

import numpy as np
import scipy.stats as stats


def pcchy(x: Union[float, np.ndarray], loc: float = 0.0, scale: float = 1.0, lower_tail: bool = True, log_p: bool = False, cdf=None) -> Union[float, np.ndarray]:
    """
    Cauchy distribution CDF.

    .. math::

        F(x) = \\frac{1}{\\pi} \\arctan\\!\\left(\\frac{x - x_0}{\\gamma}\\right) + \\frac{1}{2}

    Mirrors R's ``pcauchy(x, location, scale, lower.tail, log.p)``.

    :param x: Quantile(s).
    :param loc: Location parameter. Default 0.0.
    :param scale: Scale parameter (> 0). Default 1.0.
    :param lower_tail: If True (default), P(X <= x); else P(X > x).
    :param log_p: If True return log-probability. Default False.
    :return: CDF value(s).
    :raises ValueError: If scale <= 0.

    References
    ----------
    R Core Team (2024). pcauchy {stats}. R documentation.
    """
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}.")
    dist = stats.cauchy(loc=loc, scale=scale)
    p = dist.cdf(x) if lower_tail else dist.sf(x)
    return np.log(p) if log_p else p


def cheatsheet() -> str:
    return "pcchy({}) -> Cauchy distribution cumulative distribution function."

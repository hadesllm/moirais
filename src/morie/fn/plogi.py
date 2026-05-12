# morie.fn -- function file (hadesllm/morie)
"""Logistic distribution cumulative distribution function."""

from typing import Union

import numpy as np
import scipy.stats as stats


def plogi(x: Union[float, np.ndarray], loc: float = 0.0, scale: float = 1.0, lower_tail: bool = True, log_p: bool = False, cdf=None) -> Union[float, np.ndarray]:
    r"""
    Logistic distribution CDF.

    .. math::

        F(x) = \\frac{1}{1 + e^{-(x - \\mu)/s}}

    Mirrors R's ``plogis(x, location, scale, lower.tail, log.p)``.

    :param x: Quantile(s).
    :param loc: Location parameter. Default 0.0.
    :param scale: Scale parameter (> 0). Default 1.0.
    :param lower_tail: If True (default), P(X <= x); else P(X > x).
    :param log_p: If True return log-probability. Default False.
    :return: CDF value(s).
    :raises ValueError: If scale <= 0.

    References
    ----------
    R Core Team (2024). plogis {stats}. R documentation.
    """
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}.")
    dist = stats.logistic(loc=loc, scale=scale)
    p = dist.cdf(x) if lower_tail else dist.sf(x)
    return np.log(p) if log_p else p


def cheatsheet() -> str:
    return "plogi({}) -> Logistic distribution cumulative distribution function."

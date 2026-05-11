# morie.fn — function file (hadesllm/morie)
"""Beta distribution cumulative distribution function."""

from typing import Union

import numpy as np
import scipy.stats as stats


def pbeta(x: Union[float, np.ndarray], alpha: float, beta: float, lower_tail: bool = True, log: bool = False, cdf=None) -> Union[float, np.ndarray]:
    """
    Beta distribution CDF.

    Mirrors R's ``pbeta(x, shape1, shape2, lower.tail, log.p)``.

    :param x: Value(s) in [0, 1].
    :param alpha: First shape parameter (> 0).
    :param beta: Second shape parameter (> 0).
    :param lower_tail: If True compute P(X <= x). Default True.
    :param log: If True return log-probability. Default False.
    :return: CDF value(s).
    :raises ValueError: If alpha <= 0 or beta <= 0.

    References
    ----------
    R Core Team (2024). pbeta {stats}. R documentation.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}.")
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}.")
    dist = stats.beta(a=alpha, b=beta)
    if lower_tail:
        result = dist.logcdf(x) if log else dist.cdf(x)
    else:
        result = dist.logsf(x) if log else dist.sf(x)
    return result


def cheatsheet() -> str:
    return "pbeta({}) -> Beta distribution cumulative distribution function."

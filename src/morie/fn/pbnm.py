# morie.fn -- function file (hadesllm/morie)
"""Binomial distribution cumulative distribution function."""

from typing import Union

import numpy as np
import scipy.stats as stats


def pbinom(x: Union[int, np.ndarray], size: int, prob: float, lower_tail: bool = True, log: bool = False, cdf=None) -> Union[float, np.ndarray]:
    """
    Binomial distribution CDF.

    Mirrors R's ``pbinom(x, size, prob, lower.tail, log.p)``.

    :param x: Quantile(s).
    :param size: Number of trials (>= 0).
    :param prob: Success probability per trial (in [0, 1]).
    :param lower_tail: If True compute P(X <= x). Default True.
    :param log: If True return log-probability. Default False.
    :return: CDF value(s).
    :raises ValueError: If size < 0 or prob not in [0, 1].

    References
    ----------
    R Core Team (2024). pbinom {stats}. R documentation.
    """
    if size < 0:
        raise ValueError(f"size must be >= 0, got {size}.")
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"prob must be in [0, 1], got {prob}.")
    dist = stats.binom(n=size, p=prob)
    if lower_tail:
        result = dist.logcdf(x) if log else dist.cdf(x)
    else:
        result = dist.logsf(x) if log else dist.sf(x)
    return result


pbnm = pbinom


def cheatsheet() -> str:
    return "pbinom({}) -> Binomial distribution cumulative distribution function."

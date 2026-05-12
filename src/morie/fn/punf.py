# morie.fn -- function file (hadesllm/morie)
"""Uniform distribution cumulative distribution function."""

from typing import Union

import numpy as np
import scipy.stats as stats


def punif(x: Union[float, np.ndarray], min: float = 0.0, max: float = 1.0, lower_tail: bool = True, log: bool = False, cdf=None) -> Union[float, np.ndarray]:
    """
    Uniform distribution CDF.

    Mirrors R's ``punif(x, min, max, lower.tail, log.p)``.

    :param x: Value(s).
    :param min: Lower bound. Default 0.0.
    :param max: Upper bound. Default 1.0.
    :param lower_tail: If True compute P(X <= x). Default True.
    :param log: If True return log-probability. Default False.
    :return: CDF value(s).
    :raises ValueError: If min >= max.

    References
    ----------
    R Core Team (2024). punif {stats}. R documentation.
    """
    if min >= max:
        raise ValueError(f"min must be < max, got min={min}, max={max}.")
    dist = stats.uniform(loc=min, scale=max - min)
    if lower_tail:
        result = dist.logcdf(x) if log else dist.cdf(x)
    else:
        result = dist.logsf(x) if log else dist.sf(x)
    return result


def cheatsheet() -> str:
    return "punif({}) -> Uniform distribution cumulative distribution function."

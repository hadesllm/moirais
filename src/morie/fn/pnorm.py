# morie.fn -- function file (hadesllm/morie)
"""Normal distribution cumulative distribution function."""

from typing import Union

import numpy as np
import scipy.stats as stats


def pnorm(x: Union[float, np.ndarray], mean: float = 0.0, sd: float = 1.0, lower_tail: bool = True, log: bool = False, cdf=None) -> Union[float, np.ndarray]:
    """
    Normal distribution cumulative distribution function.

    Mirrors R's ``pnorm(x, mean, sd, lower.tail, log.p)``.

    :param x: Quantile(s).
    :param mean: Mean. Default 0.0.
    :param sd: Standard deviation (> 0). Default 1.0.
    :param lower_tail: If True (default) compute P(X <= x); else P(X > x).
    :param log: If True return log-probability. Default False.
    :return: CDF value(s).
    :raises ValueError: If sd <= 0.

    References
    ----------
    R Core Team (2024). pnorm {stats}. R documentation.
    """
    if sd <= 0:
        raise ValueError(f"sd must be > 0, got {sd}.")
    dist = stats.norm(loc=mean, scale=sd)
    if lower_tail:
        # logcdf = log(P(X <= x)) -- correct for lower tail
        result = dist.logcdf(x) if log else dist.cdf(x)
    else:
        # logsf = log(P(X > x)) = log(1 - CDF) -- correct for upper tail
        result = dist.logsf(x) if log else dist.sf(x)
    return result


def cheatsheet() -> str:
    return "pnorm({}) -> Normal distribution cumulative distribution function."

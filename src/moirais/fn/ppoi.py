# moirais.fn — function file (hadesllm/moirais)
"""Poisson distribution cumulative distribution function."""

from typing import Union

import numpy as np
import scipy.stats as stats


def ppois(x: Union[int, np.ndarray], lambda_: float, lower_tail: bool = True, log: bool = False, cdf=None) -> Union[float, np.ndarray]:
    """
    Poisson distribution CDF.

    Mirrors R's ``ppois(x, lambda, lower.tail, log.p)``.

    :param x: Quantile(s).
    :param lambda_: Rate parameter (lambda > 0).
    :param lower_tail: If True compute P(X <= x). Default True.
    :param log: If True return log-probability. Default False.
    :return: CDF value(s).
    :raises ValueError: If lambda_ <= 0.

    References
    ----------
    R Core Team (2024). ppois {stats}. R documentation.
    """
    if lambda_ <= 0:
        raise ValueError(f"lambda_ must be > 0, got {lambda_}.")
    dist = stats.poisson(mu=lambda_)
    if lower_tail:
        result = dist.logcdf(x) if log else dist.cdf(x)
    else:
        result = dist.logsf(x) if log else dist.sf(x)
    return result


def cheatsheet() -> str:
    return "ppois({}) -> Poisson distribution cumulative distribution function."

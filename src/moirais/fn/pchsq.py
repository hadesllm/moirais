# moirais.fn — function file (hadesllm/moirais)
"""Chi-squared distribution cumulative distribution function."""

from typing import Union

import numpy as np
import scipy.stats as stats


def pchisq(x: Union[float, np.ndarray], df: float, lower_tail: bool = True, log: bool = False, cdf=None) -> Union[float, np.ndarray]:
    """
    Chi-squared distribution CDF.

    Mirrors R's ``pchisq(x, df, lower.tail, log.p)``.

    :param x: Quantile(s).
    :param df: Degrees of freedom (> 0).
    :param lower_tail: If True compute P(X <= x). Default True.
    :param log: If True return log-probability. Default False.
    :return: CDF value(s).
    :raises ValueError: If df <= 0.

    References
    ----------
    R Core Team (2024). pchisq {stats}. R documentation.
    """
    if df <= 0:
        raise ValueError(f"df must be > 0, got {df}.")
    dist = stats.chi2(df=df)
    if lower_tail:
        result = dist.logcdf(x) if log else dist.cdf(x)
    else:
        result = dist.logsf(x) if log else dist.sf(x)
    return result


pchsq = pchisq


def cheatsheet() -> str:
    return "pchisq({}) -> Chi-squared distribution cumulative distribution function."

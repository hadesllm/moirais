# morie.fn -- function file (hadesllm/morie)
"""Student's t-distribution cumulative distribution function."""

from typing import Union

import numpy as np
import scipy.stats as stats


def pt(x: Union[float, np.ndarray], df: float, lower_tail: bool = True, log: bool = False, cdf=None) -> Union[float, np.ndarray]:
    """
    Student's t-distribution CDF.

    Mirrors R's ``pt(x, df, lower.tail, log.p)``.

    :param x: Quantile(s).
    :param df: Degrees of freedom (> 0).
    :param lower_tail: If True compute P(T <= x). Default True.
    :param log: If True return log-probability. Default False.
    :return: CDF value(s).
    :raises ValueError: If df <= 0.

    References
    ----------
    R Core Team (2024). pt {stats}. R documentation.
    """
    if df <= 0:
        raise ValueError(f"df must be > 0, got {df}.")
    dist = stats.t(df=df)
    if lower_tail:
        result = dist.logcdf(x) if log else dist.cdf(x)
    else:
        result = dist.logsf(x) if log else dist.sf(x)
    return result


def cheatsheet() -> str:
    return "pt({}) -> Student's t-distribution cumulative distribution function."

# morie.fn -- function file (hadesllm/morie)
"""Gamma distribution cumulative distribution function."""

from typing import Union

import numpy as np
import scipy.stats as stats


def pgamma(x: Union[float, np.ndarray], shape: float, rate: float = 1.0, scale: float | None = None, lower_tail: bool = True, log: bool = False, cdf=None) -> Union[float, np.ndarray]:
    """
    Gamma distribution CDF.

    Mirrors R's ``pgamma(x, shape, rate, scale, lower.tail, log.p)``.

    :param x: Quantile(s).
    :param shape: Shape parameter (> 0).
    :param rate: Rate parameter (> 0). Default 1.0.
    :param scale: Scale parameter (overrides rate if provided).
    :param lower_tail: If True compute P(X <= x). Default True.
    :param log: If True return log-probability. Default False.
    :return: CDF value(s).
    :raises ValueError: If shape <= 0.

    References
    ----------
    R Core Team (2024). pgamma {stats}. R documentation.
    """
    if shape <= 0:
        raise ValueError(f"shape must be > 0, got {shape}.")
    effective_scale = 1.0 / rate if scale is None else scale
    if effective_scale <= 0:
        raise ValueError(f"Effective scale must be > 0, got {effective_scale}.")
    dist = stats.gamma(a=shape, scale=effective_scale)
    if lower_tail:
        result = dist.logcdf(x) if log else dist.cdf(x)
    else:
        result = dist.logsf(x) if log else dist.sf(x)
    return result


def cheatsheet() -> str:
    return "pgamma({}) -> Gamma distribution cumulative distribution function."

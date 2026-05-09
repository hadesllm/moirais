# moirais.fn — function file (hadesllm/moirais)
"""Weibull distribution cumulative distribution function."""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.stats import weibull_min


def pweib(q: Union[float, np.ndarray], shape: float, scale: float = 1.0, lower_tail: bool = True, cdf=None) -> Union[float, np.ndarray]:
    """
    Weibull distribution cumulative distribution function.

    Mirrors R's ``pweibull(q, shape, scale, lower.tail)``.

    :param q: Quantile(s).
    :param shape: Shape parameter (> 0).
    :param scale: Scale parameter (> 0). Default 1.0.
    :param lower_tail: If True (default) return P(X <= q); else P(X > q).
    :return: CDF value(s).
    :raises ValueError: If shape <= 0 or scale <= 0.

    References
    ----------
    R Core Team (2024). pweibull {stats}. R documentation.
    """
    if shape <= 0:
        raise ValueError(f"shape must be > 0, got {shape}.")
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}.")
    dist = weibull_min(c=shape, scale=scale)
    result = dist.cdf(q) if lower_tail else dist.sf(q)
    return float(result) if np.ndim(result) == 0 else result


pweibull = pweib


def cheatsheet() -> str:
    return "pweib({}) -> Weibull distribution cumulative distribution function."

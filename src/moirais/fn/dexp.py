# moirais.fn — function file (hadesllm/moirais)
"""Exponential distribution probability density function."""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.stats import expon


def dexp(x: Union[float, np.ndarray], rate: float = 1.0) -> Union[float, np.ndarray]:
    """
    Exponential distribution probability density function.

    Mirrors R's ``dexp(x, rate)``.  Scale = 1/rate.

    :param x: Quantile(s) at which to evaluate the density.
    :param rate: Rate parameter (> 0). Default 1.0.
    :return: Density value(s).
    :raises ValueError: If rate <= 0.

    References
    ----------
    R Core Team (2024). dexp {stats}. R documentation.
    """
    if rate <= 0:
        raise ValueError(f"rate must be > 0, got {rate}.")
    result = expon.pdf(x, scale=1.0 / rate)
    return float(result) if np.ndim(result) == 0 else result


def cheatsheet() -> str:
    return "dexp({}) -> Exponential distribution probability density function."

# morie.fn — function file (hadesllm/morie)
"""Exponential distribution cumulative distribution function."""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.stats import expon


def pexp(q: Union[float, np.ndarray], rate: float = 1.0, cdf=None, *, lower_tail: bool = True) -> Union[float, np.ndarray]:
    """
    Exponential distribution cumulative distribution function.

    Mirrors R's ``pexp(q, rate, lower.tail)``.

    :param q: Quantile(s).
    :param rate: Rate parameter (> 0). Default 1.0.
    :param lower_tail: If True (default), return P(X <= q); else P(X > q).
    :return: Cumulative probability(ies).
    :raises ValueError: If rate <= 0.

    References
    ----------
    R Core Team (2024). pexp {stats}. R documentation.
    """
    if rate <= 0:
        raise ValueError(f"rate must be > 0, got {rate}.")
    result = expon.cdf(q, scale=1.0 / rate)
    if not lower_tail:
        result = 1.0 - result
    return float(result) if np.ndim(result) == 0 else result


def cheatsheet() -> str:
    return "pexp({}) -> Exponential distribution cumulative distribution function."

# morie.fn — function file (hadesllm/morie)
"""Negative binomial distribution cumulative distribution function."""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.stats import nbinom


def pnbm(q: Union[int, np.ndarray], size: float, prob: float, lower_tail: bool = True, cdf=None) -> Union[float, np.ndarray]:
    """
    Negative binomial distribution cumulative distribution function.

    Mirrors R's ``pnbinom(q, size, prob, lower.tail)``.

    :param q: Quantile(s).
    :param size: Target number of successes (> 0).
    :param prob: Probability of success (0 < prob <= 1).
    :param lower_tail: If True (default) return P(X <= q); else P(X > q).
    :return: CDF value(s).
    :raises ValueError: If size <= 0 or prob not in (0, 1].

    References
    ----------
    R Core Team (2024). pnbinom {stats}. R documentation.
    """
    if size <= 0:
        raise ValueError(f"size must be > 0, got {size}.")
    if not (0 < prob <= 1):
        raise ValueError(f"prob must be in (0, 1], got {prob}.")
    dist = nbinom(n=size, p=prob)
    result = dist.cdf(q) if lower_tail else dist.sf(q)
    return float(result) if np.ndim(result) == 0 else result


pnbinom = pnbm


def cheatsheet() -> str:
    return "pnbm({}) -> Negative binomial distribution cumulative distribution funct"

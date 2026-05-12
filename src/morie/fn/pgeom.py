# morie.fn -- function file (hadesllm/morie)
"""Geometric distribution cumulative distribution function."""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.stats import geom


def pgeom(q: Union[int, np.ndarray], prob: float, lower_tail: bool = True, cdf=None) -> Union[float, np.ndarray]:
    """
    Geometric distribution cumulative distribution function.

    Mirrors R's ``pgeom(q, prob, lower.tail)`` where q is the number
    of failures before the first success.

    :param q: Quantile(s) -- number of failures.
    :param prob: Probability of success (0 < prob <= 1).
    :param lower_tail: If True (default) return P(X <= q); else P(X > q).
    :return: CDF value(s).
    :raises ValueError: If prob not in (0, 1].

    References
    ----------
    R Core Team (2024). pgeom {stats}. R documentation.
    """
    if not (0 < prob <= 1):
        raise ValueError(f"prob must be in (0, 1], got {prob}.")
    # Shift by +1 for scipy's 1-based geom
    dist = geom(prob)
    q_shifted = np.asarray(q) + 1
    result = dist.cdf(q_shifted) if lower_tail else dist.sf(q_shifted)
    return float(result) if np.ndim(result) == 0 else result


pgeometric = pgeom


def cheatsheet() -> str:
    return "pgeom({}) -> Geometric distribution cumulative distribution function."

# morie.fn -- function file (hadesllm/morie)
"""Geometric distribution probability mass function."""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.stats import geom


def dgeom(x: Union[int, np.ndarray], prob: float) -> Union[float, np.ndarray]:
    """
    Geometric distribution probability mass function.

    Mirrors R's ``dgeom(x, prob)`` where x is the number of failures
    before the first success. scipy.stats.geom counts the trial number
    of the first success, so we shift by +1.

    :param x: Non-negative integer(s) -- number of failures before first success.
    :param prob: Probability of success (0 < prob <= 1).
    :return: PMF value(s).
    :raises ValueError: If prob not in (0, 1].

    References
    ----------
    R Core Team (2024). dgeom {stats}. R documentation.
    """
    if not (0 < prob <= 1):
        raise ValueError(f"prob must be in (0, 1], got {prob}.")
    # R's dgeom(x, prob) = prob * (1-prob)^x for x = 0, 1, 2, ...
    # scipy geom.pmf(k, p) = p * (1-p)^(k-1) for k = 1, 2, 3, ...
    # So dgeom(x, p) = geom.pmf(x + 1, p)
    result = geom.pmf(np.asarray(x) + 1, prob)
    return float(result) if np.ndim(result) == 0 else result


dgeometric = dgeom


def cheatsheet() -> str:
    return "dgeom({}) -> Geometric distribution probability mass function."

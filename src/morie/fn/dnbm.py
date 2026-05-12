# morie.fn -- function file (hadesllm/morie)
"""Negative binomial distribution probability mass function."""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.stats import nbinom


def dnbm(x: Union[int, np.ndarray], size: float, prob: float) -> Union[float, np.ndarray]:
    """
    Negative binomial distribution probability mass function.

    Mirrors R's ``dnbinom(x, size, prob)``.

    scipy parameterization: ``nbinom(n=size, p=prob)``.

    :param x: Non-negative integer value(s) (number of failures before size successes).
    :param size: Target number of successes (> 0).
    :param prob: Probability of success in each trial (0 < prob <= 1).
    :return: PMF value(s).
    :raises ValueError: If size <= 0 or prob not in (0, 1].

    References
    ----------
    R Core Team (2024). dnbinom {stats}. R documentation.
    """
    if size <= 0:
        raise ValueError(f"size must be > 0, got {size}.")
    if not (0 < prob <= 1):
        raise ValueError(f"prob must be in (0, 1], got {prob}.")
    result = nbinom.pmf(x, n=size, p=prob)
    return float(result) if np.ndim(result) == 0 else result


dnbinom = dnbm


def cheatsheet() -> str:
    return "dnbm({}) -> Negative binomial distribution probability mass function."

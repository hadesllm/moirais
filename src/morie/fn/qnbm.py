# morie.fn -- function file (hadesllm/morie)
"""Negative binomial distribution quantile function (inverse CDF)."""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.stats import nbinom


def qnbm(p: Union[float, np.ndarray], size: float, prob: float) -> Union[float, np.ndarray]:
    """
    Negative binomial distribution quantile function (inverse CDF).

    Mirrors R's ``qnbinom(p, size, prob)``.

    :param p: Probability value(s) in (0, 1).
    :param size: Target number of successes (> 0).
    :param prob: Probability of success (0 < prob <= 1).
    :return: Quantile(s).
    :raises ValueError: If size <= 0 or prob not in (0, 1].

    References
    ----------
    R Core Team (2024). qnbinom {stats}. R documentation.
    """
    if size <= 0:
        raise ValueError(f"size must be > 0, got {size}.")
    if not (0 < prob <= 1):
        raise ValueError(f"prob must be in (0, 1], got {prob}.")
    result = nbinom.ppf(p, n=size, p=prob)
    return float(result) if np.ndim(result) == 0 else result


qnbinom = qnbm


def cheatsheet() -> str:
    return "qnbm({}) -> Negative binomial distribution quantile function (inverse CD"

# morie.fn -- function file (hadesllm/morie)
"""Binomial distribution quantile function."""

from typing import Union

import numpy as np
import scipy.stats as stats


def qbinom(
    p: Union[float, np.ndarray], size: int, prob: float, lower_tail: bool = True, log: bool = False
) -> Union[int, np.ndarray]:
    """
    Binomial distribution quantile function.

    Returns the smallest integer k such that P(X <= k) >= p.
    Mirrors R's ``qbinom(p, size, prob, lower.tail, log.p)``.

    :param p: Probability value(s) in (0, 1).
    :param size: Number of trials (>= 0).
    :param prob: Success probability per trial (in [0, 1]).
    :param lower_tail: If True p = P(X <= x). Default True.
    :param log: If True p is log-probability. Default False.
    :return: Quantile(s) as integer(s).
    :raises ValueError: If size < 0 or prob not in [0, 1].

    References
    ----------
    R Core Team (2024). qbinom {stats}. R documentation.
    """
    if size < 0:
        raise ValueError(f"size must be >= 0, got {size}.")
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"prob must be in [0, 1], got {prob}.")
    p_arr = np.asarray(p, dtype=float)
    if log:
        p_arr = np.exp(p_arr)
    if not lower_tail:
        p_arr = 1.0 - p_arr
    return stats.binom(n=size, p=prob).ppf(p_arr)


qbnm = qbinom


def cheatsheet() -> str:
    return "qbinom({}) -> Binomial distribution quantile function."

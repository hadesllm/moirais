# morie.fn -- function file (hadesllm/morie)
"""Binomial distribution probability mass function."""

from typing import Union

import numpy as np
import scipy.stats as stats


def dbinom(x: Union[int, np.ndarray], size: int, prob: float, log: bool = False) -> Union[float, np.ndarray]:
    """
    Binomial distribution probability mass function.

    Computes :math:`P(X = k) = \\binom{n}{k} p^k (1-p)^{n-k}`.

    Mirrors R's ``dbinom(x, size, prob, log)``.

    :param x: Number of successes (integer >= 0).
    :param size: Number of trials (n >= 0).
    :param prob: Success probability per trial (in [0, 1]).
    :param log: If True return log-PMF. Default False.
    :return: PMF value(s).
    :raises ValueError: If size < 0 or prob not in [0, 1].

    References
    ----------
    R Core Team (2024). dbinom {stats}. R documentation.
    """
    if size < 0:
        raise ValueError(f"size must be >= 0, got {size}.")
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"prob must be in [0, 1], got {prob}.")
    dist = stats.binom(n=size, p=prob)
    return dist.logpmf(x) if log else dist.pmf(x)


dbnm = dbinom


def cheatsheet() -> str:
    return "dbinom({}) -> Binomial distribution probability mass function."

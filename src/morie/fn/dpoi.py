# morie.fn — function file (hadesllm/morie)
"""Poisson distribution probability mass function."""

from typing import Union

import numpy as np
import scipy.stats as stats


def dpois(x: Union[int, np.ndarray], lambda_: float, log: bool = False) -> Union[float, np.ndarray]:
    """
    Poisson distribution probability mass function.

    Computes :math:`P(X = k) = e^{-\\lambda} \\lambda^k / k!`.

    Mirrors R's ``dpois(x, lambda, log)``.

    :param x: Non-negative integer value(s).
    :param lambda_: Rate parameter (lambda > 0).
    :param log: If True return log-PMF. Default False.
    :return: PMF value(s).
    :raises ValueError: If lambda_ <= 0.

    References
    ----------
    R Core Team (2024). dpois {stats}. R documentation.
    """
    if lambda_ <= 0:
        raise ValueError(f"lambda_ must be > 0, got {lambda_}.")
    dist = stats.poisson(mu=lambda_)
    return dist.logpmf(x) if log else dist.pmf(x)


def cheatsheet() -> str:
    return "dpois({}) -> Poisson distribution probability mass function."

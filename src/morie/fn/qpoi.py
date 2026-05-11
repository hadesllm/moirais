# morie.fn — function file (hadesllm/morie)
"""Poisson distribution quantile function."""

from typing import Union

import numpy as np
import scipy.stats as stats


def qpois(
    p: Union[float, np.ndarray], lambda_: float, lower_tail: bool = True, log: bool = False
) -> Union[int, np.ndarray]:
    """
    Poisson distribution quantile function.

    Mirrors R's ``qpois(p, lambda, lower.tail, log.p)``.

    :param p: Probability value(s) in (0, 1).
    :param lambda_: Rate parameter (lambda > 0).
    :param lower_tail: If True p = P(X <= x). Default True.
    :param log: If True p is log-probability. Default False.
    :return: Quantile(s).
    :raises ValueError: If lambda_ <= 0.

    References
    ----------
    R Core Team (2024). qpois {stats}. R documentation.
    """
    if lambda_ <= 0:
        raise ValueError(f"lambda_ must be > 0, got {lambda_}.")
    p_arr = np.asarray(p, dtype=float)
    if log:
        p_arr = np.exp(p_arr)
    if not lower_tail:
        p_arr = 1.0 - p_arr
    return stats.poisson(mu=lambda_).ppf(p_arr)


def cheatsheet() -> str:
    return "qpois({}) -> Poisson distribution quantile function."

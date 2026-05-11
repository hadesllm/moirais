# morie.fn — function file (hadesllm/morie)
"""Beta distribution quantile function."""

from typing import Union

import numpy as np
import scipy.stats as stats


def qbeta(
    p: Union[float, np.ndarray], alpha: float, beta: float, lower_tail: bool = True, log: bool = False
) -> Union[float, np.ndarray]:
    """
    Beta distribution quantile function.

    Mirrors R's ``qbeta(p, shape1, shape2, lower.tail, log.p)``.

    :param p: Probability value(s) in (0, 1).
    :param alpha: First shape parameter (> 0).
    :param beta: Second shape parameter (> 0).
    :param lower_tail: If True p = P(X <= x). Default True.
    :param log: If True p is log-probability. Default False.
    :return: Quantile(s).
    :raises ValueError: If alpha <= 0 or beta <= 0.

    References
    ----------
    R Core Team (2024). qbeta {stats}. R documentation.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}.")
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}.")
    p_arr = np.asarray(p, dtype=float)
    if log:
        p_arr = np.exp(p_arr)
    if not lower_tail:
        p_arr = 1.0 - p_arr
    return stats.beta(a=alpha, b=beta).ppf(p_arr)


def cheatsheet() -> str:
    return "qbeta({}) -> Beta distribution quantile function."

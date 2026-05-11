# morie.fn — function file (hadesllm/morie)
"""Beta distribution probability density function."""

from typing import Union

import numpy as np
import scipy.stats as stats


def dbeta(x: Union[float, np.ndarray], alpha: float, beta: float, log: bool = False) -> Union[float, np.ndarray]:
    """
    Beta distribution probability density function.

    Mirrors R's ``dbeta(x, shape1, shape2, log)``.

    :param x: Value(s) in [0, 1].
    :param alpha: First shape parameter (> 0).
    :param beta: Second shape parameter (> 0).
    :param log: If True return log-density. Default False.
    :return: Density value(s).
    :raises ValueError: If alpha <= 0 or beta <= 0.

    References
    ----------
    R Core Team (2024). dbeta {stats}. R documentation.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}.")
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}.")
    dist = stats.beta(a=alpha, b=beta)
    return dist.logpdf(x) if log else dist.pdf(x)


def cheatsheet() -> str:
    return "dbeta({}) -> Beta distribution probability density function."

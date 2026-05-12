# morie.fn -- function file (hadesllm/morie)
"""Lognormal distribution cumulative distribution function."""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.stats import lognorm


def plnrm(q: Union[float, np.ndarray], meanlog: float = 0.0, sdlog: float = 1.0, lower_tail: bool = True, cdf=None) -> Union[float, np.ndarray]:
    """
    Lognormal distribution cumulative distribution function.

    Mirrors R's ``plnorm(q, meanlog, sdlog, lower.tail)``.

    :param q: Quantile(s).
    :param meanlog: Mean of the log. Default 0.0.
    :param sdlog: Standard deviation of the log (> 0). Default 1.0.
    :param lower_tail: If True (default) return P(X <= q); else P(X > q).
    :return: CDF value(s).
    :raises ValueError: If sdlog <= 0.

    References
    ----------
    R Core Team (2024). plnorm {stats}. R documentation.
    """
    if sdlog <= 0:
        raise ValueError(f"sdlog must be > 0, got {sdlog}.")
    dist = lognorm(s=sdlog, scale=np.exp(meanlog))
    result = dist.cdf(q) if lower_tail else dist.sf(q)
    return float(result) if np.ndim(result) == 0 else result


plnorm = plnrm


def cheatsheet() -> str:
    return "plnrm({}) -> Lognormal distribution cumulative distribution function."

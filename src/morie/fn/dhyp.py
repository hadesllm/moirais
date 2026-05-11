# morie.fn — function file (hadesllm/morie)
"""Hypergeometric distribution probability mass function."""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.stats import hypergeom


def dhyp(x: Union[int, np.ndarray], m: int, n: int, k: int) -> Union[float, np.ndarray]:
    """
    Hypergeometric distribution probability mass function.

    Mirrors R's ``dhyper(x, m, n, k)`` where:
    - m = number of success states (white balls)
    - n = number of failure states (black balls)
    - k = number of draws

    scipy parameterization: ``hypergeom(M=m+n, n=m, N=k)``.

    :param x: Non-negative integer(s) — number of successes drawn.
    :param m: Number of success states in population (>= 0).
    :param n: Number of failure states in population (>= 0).
    :param k: Number of draws (>= 0, <= m + n).
    :return: PMF value(s).
    :raises ValueError: If any parameter is negative or k > m + n.

    References
    ----------
    R Core Team (2024). dhyper {stats}. R documentation.
    """
    if m < 0:
        raise ValueError(f"m must be >= 0, got {m}.")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}.")
    if k < 0 or k > m + n:
        raise ValueError(f"k must be in [0, m+n], got {k}.")
    # scipy: hypergeom(M=total, n=success_states, N=draws)
    result = hypergeom.pmf(x, M=m + n, n=m, N=k)
    return float(result) if np.ndim(result) == 0 else result


dhyper = dhyp


def cheatsheet() -> str:
    return "dhyp({}) -> Hypergeometric distribution probability mass function."

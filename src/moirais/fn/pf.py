# moirais.fn — function file (hadesllm/moirais)
"""F-distribution cumulative distribution function."""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.stats import f as f_dist


def pf(q: Union[float, np.ndarray], dfn: float, dfd: float, cdf=None, *, lower_tail: bool = True) -> Union[float, np.ndarray]:
    """
    F-distribution cumulative distribution function.

    Mirrors R's ``pf(q, df1, df2, lower.tail)``.

    :param q: Quantile(s).
    :param dfn: Numerator degrees of freedom (> 0).
    :param dfd: Denominator degrees of freedom (> 0).
    :param lower_tail: If True (default), return P(X <= q); else P(X > q).
    :return: Cumulative probability(ies).
    :raises ValueError: If dfn <= 0 or dfd <= 0.

    References
    ----------
    R Core Team (2024). pf {stats}. R documentation.
    """
    if dfn <= 0 or dfd <= 0:
        raise ValueError(f"Degrees of freedom must be positive, got dfn={dfn}, dfd={dfd}.")
    result = f_dist.cdf(q, dfn, dfd)
    if not lower_tail:
        result = 1.0 - result
    return float(result) if np.ndim(result) == 0 else result


def cheatsheet() -> str:
    return "pf({}) -> F-distribution cumulative distribution function."

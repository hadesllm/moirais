# morie.fn — function file (hadesllm/morie)
"""F-distribution probability density function."""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.stats import f as f_dist


def df_dist(x: Union[float, np.ndarray], dfn: float, dfd: float) -> Union[float, np.ndarray]:
    """
    F-distribution probability density function.

    Mirrors R's ``df(x, df1, df2)``.  Named ``df_dist`` to avoid shadowing
    Python's built-in ``df`` abbreviation common in pandas code.

    :param x: Quantile(s) at which to evaluate the density.
    :param dfn: Numerator degrees of freedom (> 0).
    :param dfd: Denominator degrees of freedom (> 0).
    :return: Density value(s).
    :raises ValueError: If dfn <= 0 or dfd <= 0.

    References
    ----------
    R Core Team (2024). df {stats}. R documentation.
    """
    if dfn <= 0 or dfd <= 0:
        raise ValueError(f"Degrees of freedom must be positive, got dfn={dfn}, dfd={dfd}.")
    result = f_dist.pdf(x, dfn, dfd)
    return float(result) if np.ndim(result) == 0 else result


df_ = df_dist


def cheatsheet() -> str:
    return "df_dist({}) -> F-distribution probability density function."

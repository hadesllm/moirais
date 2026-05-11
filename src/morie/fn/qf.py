# morie.fn — function file (hadesllm/morie)
"""F-distribution quantile function."""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.stats import f as f_dist


def qf(p: Union[float, np.ndarray], dfn: float, dfd: float) -> Union[float, np.ndarray]:
    """
    F-distribution quantile (inverse CDF).

    Mirrors R's ``qf(p, df1, df2)``.

    :param p: Probability(ies) in [0, 1].
    :param dfn: Numerator degrees of freedom (> 0).
    :param dfd: Denominator degrees of freedom (> 0).
    :return: Quantile(s).
    :raises ValueError: If dfn <= 0 or dfd <= 0.

    References
    ----------
    R Core Team (2024). qf {stats}. R documentation.
    """
    if dfn <= 0 or dfd <= 0:
        raise ValueError(f"Degrees of freedom must be positive, got dfn={dfn}, dfd={dfd}.")
    result = f_dist.ppf(p, dfn, dfd)
    return float(result) if np.ndim(result) == 0 else result


def cheatsheet() -> str:
    return "qf({}) -> F-distribution quantile function."

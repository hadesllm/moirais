# morie.fn -- function file (hadesllm/morie)
"""Chi-squared distribution quantile function."""

from typing import Union

import numpy as np
import scipy.stats as stats


def qchisq(
    p: Union[float, np.ndarray], df: float, lower_tail: bool = True, log: bool = False
) -> Union[float, np.ndarray]:
    """
    Chi-squared distribution quantile function.

    Mirrors R's ``qchisq(p, df, lower.tail, log.p)``.

    :param p: Probability value(s) in (0, 1).
    :param df: Degrees of freedom (> 0).
    :param lower_tail: If True p = P(X <= x). Default True.
    :param log: If True p is log-probability. Default False.
    :return: Quantile(s).
    :raises ValueError: If df <= 0.

    References
    ----------
    R Core Team (2024). qchisq {stats}. R documentation.
    """
    if df <= 0:
        raise ValueError(f"df must be > 0, got {df}.")
    p_arr = np.asarray(p, dtype=float)
    if log:
        p_arr = np.exp(p_arr)
    if not lower_tail:
        p_arr = 1.0 - p_arr
    return stats.chi2(df=df).ppf(p_arr)


qchsq = qchisq


def cheatsheet() -> str:
    return "qchisq({}) -> Chi-squared distribution quantile function."

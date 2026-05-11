# morie.fn — function file (hadesllm/morie)
"""Student's t-distribution quantile function."""

from typing import Union

import numpy as np
import scipy.stats as stats


def qt(p: Union[float, np.ndarray], df: float, lower_tail: bool = True, log: bool = False) -> Union[float, np.ndarray]:
    """
    Student's t-distribution quantile function.

    Mirrors R's ``qt(p, df, lower.tail, log.p)``.

    :param p: Probability value(s) in (0, 1).
    :param df: Degrees of freedom (> 0).
    :param lower_tail: If True p = P(T <= x). Default True.
    :param log: If True, p is log-probability. Default False.
    :return: Quantile(s).
    :raises ValueError: If df <= 0.

    References
    ----------
    R Core Team (2024). qt {stats}. R documentation.
    """
    if df <= 0:
        raise ValueError(f"df must be > 0, got {df}.")
    p_arr = np.asarray(p, dtype=float)
    if log:
        p_arr = np.exp(p_arr)
    if not lower_tail:
        p_arr = 1.0 - p_arr
    return stats.t(df=df).ppf(p_arr)


def cheatsheet() -> str:
    return "qt({}) -> Student's t-distribution quantile function."

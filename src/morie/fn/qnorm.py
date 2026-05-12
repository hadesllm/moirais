# morie.fn -- function file (hadesllm/morie)
"""Normal distribution quantile function (inverse CDF)."""

from typing import Union

import numpy as np
import scipy.stats as stats


def qnorm(
    p: Union[float, np.ndarray], mean: float = 0.0, sd: float = 1.0, lower_tail: bool = True, log: bool = False
) -> Union[float, np.ndarray]:
    """
    Normal distribution quantile function (inverse CDF).

    Mirrors R's ``qnorm(p, mean, sd, lower.tail, log.p)``.

    :param p: Probability value(s) in (0, 1); or log-probabilities if log=True.
    :param mean: Mean. Default 0.0.
    :param sd: Standard deviation (> 0). Default 1.0.
    :param lower_tail: If True (default), p = P(X <= x); else p = P(X > x).
    :param log: If True, p is treated as log(probability). Default False.
    :return: Quantile(s).
    :raises ValueError: If sd <= 0.

    References
    ----------
    R Core Team (2024). qnorm {stats}. R documentation.
    """
    if sd <= 0:
        raise ValueError(f"sd must be > 0, got {sd}.")
    p_arr = np.asarray(p, dtype=float)
    if log:
        p_arr = np.exp(p_arr)
    if not lower_tail:
        p_arr = 1.0 - p_arr
    return stats.norm(loc=mean, scale=sd).ppf(p_arr)


def cheatsheet() -> str:
    return "qnorm({}) -> Normal distribution quantile function (inverse CDF)."

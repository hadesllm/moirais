# morie.fn — function file (hadesllm/morie)
"""Lognormal distribution quantile function (inverse CDF)."""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.stats import lognorm


def qlnrm(p: Union[float, np.ndarray], meanlog: float = 0.0, sdlog: float = 1.0) -> Union[float, np.ndarray]:
    """
    Lognormal distribution quantile function (inverse CDF).

    Mirrors R's ``qlnorm(p, meanlog, sdlog)``.

    :param p: Probability value(s) in (0, 1).
    :param meanlog: Mean of the log. Default 0.0.
    :param sdlog: Standard deviation of the log (> 0). Default 1.0.
    :return: Quantile(s).
    :raises ValueError: If sdlog <= 0.

    References
    ----------
    R Core Team (2024). qlnorm {stats}. R documentation.
    """
    if sdlog <= 0:
        raise ValueError(f"sdlog must be > 0, got {sdlog}.")
    result = lognorm.ppf(p, s=sdlog, scale=np.exp(meanlog))
    return float(result) if np.ndim(result) == 0 else result


qlnorm = qlnrm


def cheatsheet() -> str:
    return "qlnrm({}) -> Lognormal distribution quantile function (inverse CDF)."

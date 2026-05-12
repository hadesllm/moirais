# morie.fn -- function file (hadesllm/morie)
"""Lognormal distribution probability density function."""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.stats import lognorm


def dlnrm(x: Union[float, np.ndarray], meanlog: float = 0.0, sdlog: float = 1.0) -> Union[float, np.ndarray]:
    """
    Lognormal distribution probability density function.

    Mirrors R's ``dlnorm(x, meanlog, sdlog)``.

    scipy parameterization: ``lognorm(s=sdlog, scale=exp(meanlog))``.

    :param x: Quantile(s) at which to evaluate the density.
    :param meanlog: Mean of the log. Default 0.0.
    :param sdlog: Standard deviation of the log (> 0). Default 1.0.
    :return: Density value(s).
    :raises ValueError: If sdlog <= 0.

    References
    ----------
    R Core Team (2024). dlnorm {stats}. R documentation.
    """
    if sdlog <= 0:
        raise ValueError(f"sdlog must be > 0, got {sdlog}.")
    result = lognorm.pdf(x, s=sdlog, scale=np.exp(meanlog))
    return float(result) if np.ndim(result) == 0 else result


dlnorm = dlnrm


def cheatsheet() -> str:
    return "dlnrm({}) -> Lognormal distribution probability density function."

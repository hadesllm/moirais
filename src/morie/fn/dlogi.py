# morie.fn — function file (hadesllm/morie)
"""Logistic distribution probability density function."""

from typing import Union

import numpy as np
import scipy.stats as stats


def dlogi(
    x: Union[float, np.ndarray], loc: float = 0.0, scale: float = 1.0, log: bool = False
) -> Union[float, np.ndarray]:
    """
    Logistic distribution probability density function.

    .. math::

        f(x) = \\frac{e^{-(x-\\mu)/s}}{s\\left(1 + e^{-(x-\\mu)/s}\\right)^2}

    Mirrors R's ``dlogis(x, location, scale, log)``.

    :param x: Quantile(s) at which to evaluate the density.
    :param loc: Location parameter (mu). Default 0.0.
    :param scale: Scale parameter (s > 0). Default 1.0.
    :param log: If True return log-density. Default False.
    :return: Density value(s).
    :raises ValueError: If scale <= 0.

    References
    ----------
    R Core Team (2024). dlogis {stats}. R documentation.
    """
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}.")
    dist = stats.logistic(loc=loc, scale=scale)
    return dist.logpdf(x) if log else dist.pdf(x)


def cheatsheet() -> str:
    return "dlogi({}) -> Logistic distribution probability density function."

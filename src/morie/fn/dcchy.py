# morie.fn — function file (hadesllm/morie)
"""Cauchy distribution probability density function."""

from typing import Union

import numpy as np
import scipy.stats as stats


def dcchy(
    x: Union[float, np.ndarray], loc: float = 0.0, scale: float = 1.0, log: bool = False
) -> Union[float, np.ndarray]:
    """
    Cauchy distribution probability density function.

    .. math::

        f(x) = \\frac{1}{\\pi \\gamma \\left[1 + \\left(\\frac{x - x_0}{\\gamma}\\right)^2\\right]}

    Mirrors R's ``dcauchy(x, location, scale, log)``.

    :param x: Quantile(s) at which to evaluate the density.
    :param loc: Location parameter (x0). Default 0.0.
    :param scale: Scale parameter (gamma > 0). Default 1.0.
    :param log: If True return log-density. Default False.
    :return: Density value(s).
    :raises ValueError: If scale <= 0.

    References
    ----------
    R Core Team (2024). dcauchy {stats}. R documentation.
    """
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}.")
    dist = stats.cauchy(loc=loc, scale=scale)
    return dist.logpdf(x) if log else dist.pdf(x)


def cheatsheet() -> str:
    return "dcchy({}) -> Cauchy distribution probability density function."

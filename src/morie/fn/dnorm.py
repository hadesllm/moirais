# morie.fn -- function file (hadesllm/morie)
"""Normal distribution probability density function."""

from typing import Union

import numpy as np
import scipy.stats as stats


def dnorm(
    x: Union[float, np.ndarray], mean: float = 0.0, sd: float = 1.0, log: bool = False
) -> Union[float, np.ndarray]:
    r"""
    Normal distribution probability density function.

    Computes the density of :math:`X \\sim \\mathcal{N}(\\mu, \\sigma^2)`:

    .. math::

        f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}}
               \\exp\\!\\left(-\\frac{(x - \\mu)^2}{2\\sigma^2}\\right)

    Mirrors R's ``dnorm(x, mean, sd, log)``.

    :param x: Quantile(s) at which to evaluate the density.
    :param mean: Mean of the distribution (mu). Default 0.0.
    :param sd: Standard deviation (sigma > 0). Default 1.0.
    :param log: If True return log-density (for numerical stability). Default False.
    :return: Density value(s).
    :raises ValueError: If sd <= 0.

    References
    ----------
    R Core Team (2024). dnorm {stats}. R documentation.
    """
    if sd <= 0:
        raise ValueError(f"sd must be > 0, got {sd}.")
    dist = stats.norm(loc=mean, scale=sd)
    result = dist.logpdf(x) if log else dist.pdf(x)
    return result


def cheatsheet() -> str:
    return "dnorm({}) -> Normal distribution probability density function."

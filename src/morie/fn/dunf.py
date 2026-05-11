# morie.fn — function file (hadesllm/morie)
"""Uniform distribution probability density function."""

from typing import Union

import numpy as np
import scipy.stats as stats


def dunif(
    x: Union[float, np.ndarray], min: float = 0.0, max: float = 1.0, log: bool = False
) -> Union[float, np.ndarray]:
    """
    Uniform distribution probability density function.

    Density is 1/(max - min) for x in [min, max], 0 otherwise.
    Mirrors R's ``dunif(x, min, max, log)``.

    :param x: Value(s).
    :param min: Lower bound of the support. Default 0.0.
    :param max: Upper bound of the support. Default 1.0.
    :param log: If True return log-density. Default False.
    :return: Density value(s).
    :raises ValueError: If min >= max.

    References
    ----------
    R Core Team (2024). dunif {stats}. R documentation.
    """
    if min >= max:
        raise ValueError(f"min must be < max, got min={min}, max={max}.")
    dist = stats.uniform(loc=min, scale=max - min)
    return dist.logpdf(x) if log else dist.pdf(x)


def cheatsheet() -> str:
    return "dunif({}) -> Uniform distribution probability density function."

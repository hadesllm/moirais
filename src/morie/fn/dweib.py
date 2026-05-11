# morie.fn — function file (hadesllm/morie)
"""Weibull distribution probability density function."""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.stats import weibull_min


def dweib(x: Union[float, np.ndarray], shape: float, scale: float = 1.0) -> Union[float, np.ndarray]:
    """
    Weibull distribution probability density function.

    Mirrors R's ``dweibull(x, shape, scale)``.

    :param x: Quantile(s) at which to evaluate the density.
    :param shape: Shape parameter (> 0).
    :param scale: Scale parameter (> 0). Default 1.0.
    :return: Density value(s).
    :raises ValueError: If shape <= 0 or scale <= 0.

    References
    ----------
    R Core Team (2024). dweibull {stats}. R documentation.
    """
    if shape <= 0:
        raise ValueError(f"shape must be > 0, got {shape}.")
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}.")
    result = weibull_min.pdf(x, c=shape, scale=scale)
    return float(result) if np.ndim(result) == 0 else result


dweibull = dweib


def cheatsheet() -> str:
    return "dweib({}) -> Weibull distribution probability density function."

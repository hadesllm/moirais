# morie.fn — function file (hadesllm/morie)
"""Gamma distribution probability density function."""

from typing import Union

import numpy as np
import scipy.stats as stats


def dgamma(
    x: Union[float, np.ndarray], shape: float, rate: float = 1.0, scale: float | None = None, log: bool = False
) -> Union[float, np.ndarray]:
    """
    Gamma distribution probability density function.

    Supports both rate and scale parameterisation (rate = 1/scale).
    When ``scale`` is provided it takes precedence over ``rate``.
    Mirrors R's ``dgamma(x, shape, rate, scale, log)``.

    :param x: Quantile(s) (>= 0).
    :param shape: Shape parameter alpha (> 0).
    :param rate: Rate parameter beta = 1/scale (> 0). Default 1.0.
    :param scale: Scale parameter theta = 1/rate. Overrides rate if provided.
    :param log: If True return log-density. Default False.
    :return: Density value(s).
    :raises ValueError: If shape <= 0, or both rate and scale are inconsistent.

    References
    ----------
    R Core Team (2024). dgamma {stats}. R documentation.
    """
    if shape <= 0:
        raise ValueError(f"shape must be > 0, got {shape}.")
    effective_scale = 1.0 / rate if scale is None else scale
    if effective_scale <= 0:
        raise ValueError(f"Effective scale must be > 0, got {effective_scale}.")
    dist = stats.gamma(a=shape, scale=effective_scale)
    return dist.logpdf(x) if log else dist.pdf(x)


def cheatsheet() -> str:
    return "dgamma({}) -> Gamma distribution probability density function."

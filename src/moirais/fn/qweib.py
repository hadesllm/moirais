# moirais.fn — function file (hadesllm/moirais)
"""Weibull distribution quantile function (inverse CDF)."""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.stats import weibull_min


def qweib(p: Union[float, np.ndarray], shape: float, scale: float = 1.0) -> Union[float, np.ndarray]:
    """
    Weibull distribution quantile function (inverse CDF).

    Mirrors R's ``qweibull(p, shape, scale)``.

    :param p: Probability value(s) in (0, 1).
    :param shape: Shape parameter (> 0).
    :param scale: Scale parameter (> 0). Default 1.0.
    :return: Quantile(s).
    :raises ValueError: If shape <= 0 or scale <= 0.

    References
    ----------
    R Core Team (2024). qweibull {stats}. R documentation.
    """
    if shape <= 0:
        raise ValueError(f"shape must be > 0, got {shape}.")
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}.")
    result = weibull_min.ppf(p, c=shape, scale=scale)
    return float(result) if np.ndim(result) == 0 else result


qweibull = qweib


def cheatsheet() -> str:
    return "qweib({}) -> Weibull distribution quantile function (inverse CDF)."

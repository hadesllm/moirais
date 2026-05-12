# morie.fn -- function file (hadesllm/morie)
"""Draw a random sample from a Weibull distribution."""

from __future__ import annotations

import numpy as np


def rweib(n: int, shape: float, scale: float = 1.0, seed: int = 42) -> np.ndarray:
    """
    Draw a random sample from a Weibull distribution.

    Mirrors R's ``rweibull(n, shape, scale)``.

    :param n: Number of observations to draw (> 0).
    :param shape: Shape parameter (> 0).
    :param scale: Scale parameter (> 0). Default 1.0.
    :param seed: Random seed for reproducibility. Default 42.
    :return: 1-D array of length n.
    :raises ValueError: If n <= 0, shape <= 0, or scale <= 0.

    References
    ----------
    R Core Team (2024). rweibull {stats}. R documentation.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}.")
    if shape <= 0:
        raise ValueError(f"shape must be > 0, got {shape}.")
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}.")
    rng = np.random.default_rng(seed)
    return scale * rng.weibull(shape, size=n)


rweibull = rweib


def cheatsheet() -> str:
    return "rweib({}) -> Draw a random sample from a Weibull distribution."

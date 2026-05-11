# morie.fn — function file (hadesllm/morie)
"""Draw a random sample from a negative binomial distribution."""

from __future__ import annotations

import numpy as np


def rnbm(n: int, size: int, prob: float, seed: int = 42) -> np.ndarray:
    """
    Draw a random sample from a negative binomial distribution.

    Mirrors R's ``rnbinom(n, size, prob)``.

    :param n: Number of observations to draw (> 0).
    :param size: Target number of successes (> 0).
    :param prob: Probability of success (0 < prob <= 1).
    :param seed: Random seed for reproducibility. Default 42.
    :return: 1-D array of length n.
    :raises ValueError: If n <= 0, size <= 0, or prob not in (0, 1].

    References
    ----------
    R Core Team (2024). rnbinom {stats}. R documentation.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}.")
    if size <= 0:
        raise ValueError(f"size must be > 0, got {size}.")
    if not (0 < prob <= 1):
        raise ValueError(f"prob must be in (0, 1], got {prob}.")
    rng = np.random.default_rng(seed)
    return rng.negative_binomial(n=size, p=prob, size=n)


rnbinom = rnbm


def cheatsheet() -> str:
    return "rnbm({}) -> Draw a random sample from a negative binomial distribution."

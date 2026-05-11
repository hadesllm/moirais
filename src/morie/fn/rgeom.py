# morie.fn — function file (hadesllm/morie)
"""Draw a random sample from a geometric distribution."""

from __future__ import annotations

import numpy as np


def rgeom(n: int, prob: float, seed: int = 42) -> np.ndarray:
    """
    Draw a random sample from a geometric distribution.

    Mirrors R's ``rgeom(n, prob)`` — returns number of failures
    before first success.

    :param n: Number of observations to draw (> 0).
    :param prob: Probability of success (0 < prob <= 1).
    :param seed: Random seed for reproducibility. Default 42.
    :return: 1-D array of length n.
    :raises ValueError: If n <= 0 or prob not in (0, 1].

    References
    ----------
    R Core Team (2024). rgeom {stats}. R documentation.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}.")
    if not (0 < prob <= 1):
        raise ValueError(f"prob must be in (0, 1], got {prob}.")
    rng = np.random.default_rng(seed)
    return rng.geometric(p=prob, size=n) - 1  # numpy returns 1-based, R is 0-based


rgeometric = rgeom


def cheatsheet() -> str:
    return "rgeom({}) -> Draw a random sample from a geometric distribution."

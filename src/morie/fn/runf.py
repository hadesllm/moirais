# morie.fn — function file (hadesllm/morie)
"""Draw a random sample from a uniform distribution."""

import numpy as np


def runif(n: int, min: float = 0.0, max: float = 1.0, seed: int | None = None) -> np.ndarray:
    """
    Draw a random sample from a uniform distribution.

    Mirrors R's ``runif(n, min, max)``.

    :param n: Number of observations to draw (> 0).
    :param min: Lower bound. Default 0.0.
    :param max: Upper bound. Default 1.0.
    :param seed: Random seed for reproducibility. Default None.
    :return: 1-D array of length n.
    :raises ValueError: If n <= 0 or min >= max.

    References
    ----------
    R Core Team (2024). runif {stats}. R documentation.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}.")
    if min >= max:
        raise ValueError(f"min must be < max, got min={min}, max={max}.")
    rng = np.random.default_rng(seed)
    return rng.uniform(low=min, high=max, size=n)


def cheatsheet() -> str:
    return "runif({}) -> Draw a random sample from a uniform distribution."

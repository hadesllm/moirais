# moirais.fn — function file (hadesllm/moirais)
"""Draw a random sample from a normal distribution."""

import numpy as np


def rnorm(n: int, mean: float = 0.0, sd: float = 1.0, seed: int | None = None) -> np.ndarray:
    """
    Draw a random sample from a normal distribution.

    Mirrors R's ``rnorm(n, mean, sd)``.

    :param n: Number of observations to draw (> 0).
    :param mean: Mean. Default 0.0.
    :param sd: Standard deviation (> 0). Default 1.0.
    :param seed: Random seed for reproducibility. Default None.
    :return: 1-D array of length n.
    :raises ValueError: If n <= 0 or sd <= 0.

    References
    ----------
    R Core Team (2024). rnorm {stats}. R documentation.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}.")
    if sd <= 0:
        raise ValueError(f"sd must be > 0, got {sd}.")
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean, scale=sd, size=n)


def cheatsheet() -> str:
    return "rnorm({}) -> Draw a random sample from a normal distribution."

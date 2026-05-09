# moirais.fn — function file (hadesllm/moirais)
"""Draw a random sample from a binomial distribution."""

import numpy as np


def rbinom(n: int, size: int, prob: float, seed: int | None = None) -> np.ndarray:
    """
    Draw a random sample from a binomial distribution.

    Mirrors R's ``rbinom(n, size, prob)``.

    :param n: Number of observations to draw (> 0).
    :param size: Number of trials per observation (>= 0).
    :param prob: Success probability per trial (in [0, 1]).
    :param seed: Random seed for reproducibility. Default None.
    :return: 1-D integer array of length n.
    :raises ValueError: If n <= 0, size < 0, or prob not in [0, 1].

    References
    ----------
    R Core Team (2024). rbinom {stats}. R documentation.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}.")
    if size < 0:
        raise ValueError(f"size must be >= 0, got {size}.")
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"prob must be in [0, 1], got {prob}.")
    rng = np.random.default_rng(seed)
    return rng.binomial(n=size, p=prob, size=n)


rbnm = rbinom


def cheatsheet() -> str:
    return "rbinom({}) -> Draw a random sample from a binomial distribution."

# morie.fn -- function file (hadesllm/morie)
"""Draw a random sample from a Poisson distribution."""

import numpy as np


def rpois(n: int, lambda_: float, seed: int | None = None) -> np.ndarray:
    """
    Draw a random sample from a Poisson distribution.

    Mirrors R's ``rpois(n, lambda)``.

    :param n: Number of observations to draw (> 0).
    :param lambda_: Rate parameter (lambda > 0).
    :param seed: Random seed for reproducibility. Default None.
    :return: 1-D integer array of length n.
    :raises ValueError: If n <= 0 or lambda_ <= 0.

    References
    ----------
    R Core Team (2024). rpois {stats}. R documentation.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}.")
    if lambda_ <= 0:
        raise ValueError(f"lambda_ must be > 0, got {lambda_}.")
    rng = np.random.default_rng(seed)
    return rng.poisson(lam=lambda_, size=n)


def cheatsheet() -> str:
    return "rpois({}) -> Draw a random sample from a Poisson distribution."

# moirais.fn — function file (hadesllm/moirais)
"""Draw a random sample from a lognormal distribution."""

from __future__ import annotations

import numpy as np


def rlnrm(n: int, meanlog: float = 0.0, sdlog: float = 1.0, seed: int = 42) -> np.ndarray:
    """
    Draw a random sample from a lognormal distribution.

    Mirrors R's ``rlnorm(n, meanlog, sdlog)``.

    :param n: Number of observations to draw (> 0).
    :param meanlog: Mean of the log. Default 0.0.
    :param sdlog: Standard deviation of the log (> 0). Default 1.0.
    :param seed: Random seed for reproducibility. Default 42.
    :return: 1-D array of length n.
    :raises ValueError: If n <= 0 or sdlog <= 0.

    References
    ----------
    R Core Team (2024). rlnorm {stats}. R documentation.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}.")
    if sdlog <= 0:
        raise ValueError(f"sdlog must be > 0, got {sdlog}.")
    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=meanlog, sigma=sdlog, size=n)


rlnorm = rlnrm


def cheatsheet() -> str:
    return "rlnrm({}) -> Draw a random sample from a lognormal distribution."

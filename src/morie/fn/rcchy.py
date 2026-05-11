# morie.fn — function file (hadesllm/morie)
"""Cauchy distribution random variate generation."""


import numpy as np
import scipy.stats as stats


def rcchy(n: int, loc: float = 0.0, scale: float = 1.0, seed: int | None = None) -> np.ndarray:
    """
    Generate random variates from a Cauchy distribution.

    Mirrors R's ``rcauchy(n, location, scale)``.

    :param n: Number of variates to generate.
    :param loc: Location parameter. Default 0.0.
    :param scale: Scale parameter (> 0). Default 1.0.
    :param seed: Random seed for reproducibility.
    :return: Array of *n* Cauchy-distributed variates.
    :raises ValueError: If scale <= 0 or n < 1.

    References
    ----------
    R Core Team (2024). rcauchy {stats}. R documentation.
    """
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}.")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    rng = np.random.default_rng(seed)
    return stats.cauchy.rvs(loc=loc, scale=scale, size=n, random_state=rng)


def cheatsheet() -> str:
    return "rcchy({}) -> Cauchy distribution random variate generation."

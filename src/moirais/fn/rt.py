# moirais.fn — function file (hadesllm/moirais)
"""Student's t-distribution random variate generation."""

from __future__ import annotations

import numpy as np


def rt(n: int, df: float, *, seed: int = 42) -> np.ndarray:
    """
    Generate random variates from Student's t-distribution.

    Mirrors R's ``rt(n, df)``.

    :param n: Number of variates to generate.
    :param df: Degrees of freedom (> 0).
    :param seed: RNG seed for reproducibility. Default 42.
    :return: Array of *n* random variates.
    :raises ValueError: If df <= 0.

    References
    ----------
    R Core Team (2024). rt {stats}. R documentation.
    """
    if df <= 0:
        raise ValueError(f"df must be > 0, got {df}.")
    return np.random.default_rng(seed).standard_t(df, size=n)


def cheatsheet() -> str:
    return "rt({}) -> Student's t-distribution random variate generation."

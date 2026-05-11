# morie.fn — function file (hadesllm/morie)
"""Chi-squared distribution random variate generation."""

from __future__ import annotations

import numpy as np


def rchisq(n: int, df: float, *, seed: int = 42) -> np.ndarray:
    """
    Generate random variates from the chi-squared distribution.

    Mirrors R's ``rchisq(n, df)``.

    :param n: Number of variates to generate.
    :param df: Degrees of freedom (> 0).
    :param seed: RNG seed for reproducibility. Default 42.
    :return: Array of *n* random variates.
    :raises ValueError: If df <= 0.

    References
    ----------
    R Core Team (2024). rchisq {stats}. R documentation.
    """
    if df <= 0:
        raise ValueError(f"df must be > 0, got {df}.")
    return np.random.default_rng(seed).chisquare(df, size=n)


def cheatsheet() -> str:
    return "rchisq({}) -> Chi-squared distribution random variate generation."

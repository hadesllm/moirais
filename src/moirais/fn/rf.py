# moirais.fn — function file (hadesllm/moirais)
"""F-distribution random variate generation."""

from __future__ import annotations

import numpy as np


def rf_dist(n: int, dfn: float, dfd: float, *, seed: int = 42) -> np.ndarray:
    """
    Generate random variates from the F-distribution.

    Mirrors R's ``rf(n, df1, df2)``.  Named ``rf_dist`` to avoid shadowing
    builtins; aliased as ``rf``.

    :param n: Number of variates to generate.
    :param dfn: Numerator degrees of freedom (> 0).
    :param dfd: Denominator degrees of freedom (> 0).
    :param seed: RNG seed for reproducibility. Default 42.
    :return: Array of *n* random variates.
    :raises ValueError: If dfn <= 0 or dfd <= 0.

    References
    ----------
    R Core Team (2024). rf {stats}. R documentation.
    """
    if dfn <= 0 or dfd <= 0:
        raise ValueError(f"Degrees of freedom must be positive, got dfn={dfn}, dfd={dfd}.")
    return np.random.default_rng(seed).f(dfn, dfd, size=n)


rf = rf_dist


def cheatsheet() -> str:
    return "rf_dist({}) -> F-distribution random variate generation."

"""Simulate ideal points from various distributions."""

from __future__ import annotations

from ._containers import DescriptiveResult


def simulate_ideal_points(n: int = 100, n_dims: int = 2, dist: str = "normal", seed: int = 42) -> DescriptiveResult:
    """Generate synthetic ideal points.

    :param n: Number of respondents.
    :param n_dims: Number of dimensions.
    :param dist: Distribution ('normal' or 'uniform').
    :param seed: Random seed.
    :return: DescriptiveResult with ideal point matrix.

    .. epigraph:: "Dattebayo!" -- Naruto Uzumaki, Naruto
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    if dist == "uniform":
        points = rng.uniform(-2, 2, (n, n_dims))
    else:
        points = rng.standard_normal((n, n_dims))
    return DescriptiveResult(
        name="simulate_ideal_points",
        value=n,
        extra={"points": points.tolist(), "n_dims": n_dims, "dist": dist},
    )


simip = simulate_ideal_points


def cheatsheet() -> str:
    return "simulate_ideal_points({}) -> Simulate ideal points from various distributions."

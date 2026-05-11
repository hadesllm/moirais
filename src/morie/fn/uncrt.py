"""Utility with uncertainty (stochastic utility model)."""

from __future__ import annotations

from ._containers import DescriptiveResult


def utility_uncertainty(ideal, position, sigma: float = 1.0, n_draws: int = 1000, seed: int = 42) -> DescriptiveResult:
    """Expected utility under Gaussian noise on position perception.

    :param ideal: Voter ideal point.
    :param position: True candidate position.
    :param sigma: Noise standard deviation.
    :param n_draws: Monte Carlo draws.
    :param seed: Random seed.
    :return: DescriptiveResult with expected utility.

    .. epigraph:: "Equivalent Exchange." -- Alphonse Elric, Fullmetal Alchemist
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    ideal = np.asarray(ideal, dtype=float).ravel()
    position = np.asarray(position, dtype=float).ravel()
    noise = rng.normal(0, sigma, size=(n_draws, len(position)))
    perceived = position + noise
    utilities = -np.sum((perceived - ideal) ** 2, axis=1)
    eu = float(np.mean(utilities))
    return DescriptiveResult(
        name="utility_uncertainty",
        value=eu,
        extra={"sigma": sigma, "std_utility": float(np.std(utilities)), "n_draws": n_draws},
    )


uncrt = utility_uncertainty


def cheatsheet() -> str:
    return "utility_uncertainty({}) -> Utility with uncertainty (stochastic utility model)."

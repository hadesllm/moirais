"""Simulate random utility shocks."""

from __future__ import annotations

from ._containers import DescriptiveResult


def simulate_utility_shocks(n=100, sigma=1.0) -> DescriptiveResult:
    """Generate random utility error components.

    .. epigraph:: If I have seen further it is by standing on the shoulders of giants. -- Isaac Newton
    """
    import numpy as np

    rng = np.random.default_rng(42)
    shocks = rng.normal(0, sigma, size=n)
    return DescriptiveResult(
        name="simulate_utility_shocks",
        value=float(np.std(shocks)),
        extra={
            "shocks": shocks,
            "n": n,
            "sigma": sigma,
            "mean": float(np.mean(shocks)),
            "std": float(np.std(shocks)),
        },
    )


smutl = simulate_utility_shocks


def cheatsheet() -> str:
    return "simulate_utility_shocks({}) -> Simulate random utility shocks."

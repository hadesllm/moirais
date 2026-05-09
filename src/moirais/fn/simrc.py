"""Simulate roll-call vote matrix."""

from __future__ import annotations

from ._containers import DescriptiveResult


def simulate_roll_call(n_leg: int = 50, n_votes: int = 100, n_dims: int = 1, seed: int = 42) -> DescriptiveResult:
    """Simulate a spatial roll-call vote matrix.

    :param n_leg: Number of legislators.
    :param n_votes: Number of roll-call votes.
    :param n_dims: Dimensionality of issue space.
    :param seed: Random seed.
    :return: DescriptiveResult with vote matrix.

    .. epigraph:: "Bankai!" -- Ichigo Kurosaki, Bleach
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    ideals = rng.standard_normal((n_leg, n_dims))
    cutting = rng.standard_normal((n_votes, n_dims))
    midpoints = rng.standard_normal((n_votes, n_dims))
    votes = np.zeros((n_leg, n_votes), dtype=int)
    for v in range(n_votes):
        proj = ideals @ cutting[v] - cutting[v] @ midpoints[v]
        votes[:, v] = (proj > 0).astype(int)
    return DescriptiveResult(
        name="simulate_roll_call",
        value=int(votes.sum()),
        extra={"votes": votes.tolist(), "n_leg": n_leg, "n_votes": n_votes, "n_dims": n_dims},
    )


simrc = simulate_roll_call


def cheatsheet() -> str:
    return "simulate_roll_call({}) -> Simulate roll-call vote matrix."

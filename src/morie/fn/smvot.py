"""Simulate spatial voting."""

from __future__ import annotations

from ._containers import DescriptiveResult


def simulate_voting_spatial(X, bills, beta=15.0) -> DescriptiveResult:
    """Simulate vote matrix from spatial model with logistic link.

    .. epigraph:: "Apply yourself." -- Walter White, Breaking Bad
    """
    import numpy as np

    rng = np.random.default_rng(42)
    X = np.asarray(X, dtype=float)
    bills = np.asarray(bills, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if bills.ndim == 1:
        bills = bills.reshape(-1, 1)
    n_leg = X.shape[0]
    n_bills = bills.shape[0]
    votes = np.zeros((n_leg, n_bills))
    for j in range(n_bills):
        dist = np.linalg.norm(X - bills[j], axis=1)
        prob_yea = 1.0 / (1.0 + np.exp(beta * (dist - 0.5)))
        votes[:, j] = (rng.uniform(size=n_leg) < prob_yea).astype(float)
    return DescriptiveResult(
        name="simulate_voting_spatial",
        value=float(np.mean(votes)),
        extra={
            "vote_matrix": votes,
            "n_legislators": n_leg,
            "n_bills": n_bills,
            "yea_rate": float(np.mean(votes)),
        },
    )


smvot = simulate_voting_spatial


def cheatsheet() -> str:
    return "simulate_voting_spatial({}) -> Simulate spatial voting."

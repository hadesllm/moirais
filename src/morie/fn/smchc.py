"""Simulate choice probabilities from utilities."""

from __future__ import annotations

from ._containers import DescriptiveResult


def simulate_choice_prob(utility_yea, utility_nay) -> DescriptiveResult:
    """Choice probabilities from yea/nay utilities via logistic.

    .. epigraph:: "Fly." -- Breaking Bad
    """
    import numpy as np

    u_yea = np.asarray(utility_yea, dtype=float)
    u_nay = np.asarray(utility_nay, dtype=float)
    diff = u_yea - u_nay
    prob_yea = 1.0 / (1.0 + np.exp(-diff))
    return DescriptiveResult(
        name="simulate_choice_prob",
        value=float(np.mean(prob_yea)),
        extra={
            "prob_yea": prob_yea,
            "mean_prob": float(np.mean(prob_yea)),
            "n": len(u_yea),
        },
    )


smchc = simulate_choice_prob


def cheatsheet() -> str:
    return "simulate_choice_prob({}) -> Simulate choice probabilities from utilities."

# morie.fn — function file (hadesllm/morie)
"""EM expectation step."""

from __future__ import annotations

from ._containers import DescriptiveResult


def em_expectation_step(theta, alpha, beta, votes) -> DescriptiveResult:
    """E-step: compute expected log-likelihood contributions.

    .. epigraph:: "I drink and I know things." -- Tyrion, Game of Thrones
    """
    import numpy as np

    theta = np.asarray(theta, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    votes = np.asarray(votes, dtype=float)
    if votes.ndim == 1:
        votes = votes.reshape(1, -1)
    n_leg, n_items = votes.shape
    Q = 0.0
    for i in range(n_leg):
        for j in range(n_items):
            if np.isnan(votes[i, j]):
                continue
            logit = alpha[j] * (theta[i] - beta[j])
            p = 1.0 / (1.0 + np.exp(-logit))
            p = np.clip(p, 1e-15, 1 - 1e-15)
            Q += votes[i, j] * np.log(p) + (1 - votes[i, j]) * np.log(1 - p)
    return DescriptiveResult(
        name="em_expectation_step",
        value=float(Q),
        extra={"Q": float(Q), "n_legislators": n_leg, "n_items": n_items},
    )


emexp = em_expectation_step


def cheatsheet() -> str:
    return "em_expectation_step({}) -> EM expectation step."

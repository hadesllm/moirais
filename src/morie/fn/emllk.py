# morie.fn — function file (hadesllm/morie)
"""EM log-likelihood computation."""

from __future__ import annotations

from ._containers import DescriptiveResult


def em_log_likelihood(votes, theta, alpha, beta) -> DescriptiveResult:
    """Compute full log-likelihood for IRT model.

    .. epigraph:: "The things I do for love." -- Jaime, Game of Thrones
    """
    import numpy as np

    votes = np.asarray(votes, dtype=float)
    theta = np.asarray(theta, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if votes.ndim == 1:
        votes = votes.reshape(1, -1)
    ll = 0.0
    for i in range(votes.shape[0]):
        for j in range(votes.shape[1]):
            if np.isnan(votes[i, j]):
                continue
            logit = alpha[j] * (theta[i] - beta[j])
            p = 1.0 / (1.0 + np.exp(-logit))
            p = np.clip(p, 1e-15, 1 - 1e-15)
            ll += votes[i, j] * np.log(p) + (1 - votes[i, j]) * np.log(1 - p)
    return DescriptiveResult(
        name="em_log_likelihood",
        value=float(ll),
        extra={
            "log_likelihood": float(ll),
            "n_legislators": votes.shape[0],
            "n_items": votes.shape[1],
        },
    )


emllk = em_log_likelihood


def cheatsheet() -> str:
    return "em_log_likelihood({}) -> EM log-likelihood computation."

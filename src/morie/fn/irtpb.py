# morie.fn -- function file (hadesllm/morie)
"""IRT probability of correct response."""

from __future__ import annotations

from ._containers import DescriptiveResult


def irt_probability(theta, alpha, beta) -> DescriptiveResult:
    """2PL IRT probability: P(Y=1|theta,alpha,beta).

    .. epigraph:: What is now proved was once only imagined. -- William Blake
    """
    import numpy as np

    theta = np.asarray(theta, dtype=float)
    alpha = float(alpha)
    beta = float(beta)
    logit = alpha * (theta - beta)
    prob = 1.0 / (1.0 + np.exp(-logit))
    return DescriptiveResult(
        name="irt_probability",
        value=float(np.mean(prob)),
        extra={
            "probabilities": prob,
            "alpha": alpha,
            "beta": beta,
            "mean_prob": float(np.mean(prob)),
            "n": len(theta.ravel()),
        },
    )


irtpb = irt_probability


def cheatsheet() -> str:
    return "irt_probability({}) -> IRT probability of correct response."

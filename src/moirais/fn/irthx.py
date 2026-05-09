# moirais.fn — function file (hadesllm/moirais)
"""Heteroskedastic IRT probabilities."""

from __future__ import annotations

from ._containers import DescriptiveResult


def irt_heteroskedastic(theta, alpha, beta, sigma_i) -> DescriptiveResult:
    """IRT with person-specific error variance (heteroskedastic).

    .. epigraph:: "A girl has no name." -- Arya, Game of Thrones
    """
    import numpy as np

    theta = np.asarray(theta, dtype=float)
    sigma_i = np.asarray(sigma_i, dtype=float)
    logit = alpha * (theta - beta) / np.maximum(sigma_i, 1e-10)
    prob = 1.0 / (1.0 + np.exp(-logit))
    return DescriptiveResult(
        name="irt_heteroskedastic",
        value=float(np.mean(prob)),
        extra={
            "probabilities": prob,
            "alpha": float(alpha),
            "beta": float(beta),
            "mean_sigma": float(np.mean(sigma_i)),
            "n": len(theta.ravel()),
        },
    )


irthx = irt_heteroskedastic


def cheatsheet() -> str:
    return "irt_heteroskedastic({}) -> Heteroskedastic IRT probabilities."

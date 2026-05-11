# morie.fn — function file (hadesllm/morie)
"""Mixture model for recidivism: recidivists vs desisters."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import DescriptiveResult


def recidivism_mixture(
    recidivism_times: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> DescriptiveResult:
    """Mixture model separating recidivists from desisters.

    EM algorithm fitting a two-component exponential mixture.

    Parameters
    ----------
    recidivism_times : ndarray
        Times to recidivism (positive values only, excluding censored).
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    DescriptiveResult
        extra: mix_prob, lambda1, lambda2.
    """
    t = np.asarray(recidivism_times, dtype=float)
    t = t[t > 0]
    n = len(t)
    if n < 4:
        return DescriptiveResult(
            name="recidivism_mixture",
            value=None,
            extra={"mix_prob": float("nan"), "lambda1": float("nan"), "lambda2": float("nan")},
        )
    pi = 0.5
    lam1 = 1.0 / np.percentile(t, 25)
    lam2 = 1.0 / np.percentile(t, 75)
    for _ in range(max_iter):
        d1 = pi * lam1 * np.exp(-lam1 * t)
        d2 = (1 - pi) * lam2 * np.exp(-lam2 * t)
        total = d1 + d2
        total = np.maximum(total, 1e-300)
        gamma = d1 / total
        pi_new = np.mean(gamma)
        lam1_new = np.sum(gamma) / np.sum(gamma * t) if np.sum(gamma * t) > 0 else lam1
        lam2_new = np.sum(1 - gamma) / np.sum((1 - gamma) * t) if np.sum((1 - gamma) * t) > 0 else lam2
        if abs(pi_new - pi) < tol:
            break
        pi, lam1, lam2 = pi_new, lam1_new, lam2_new
    return DescriptiveResult(
        name="recidivism_mixture",
        value=float(pi),
        extra={"mix_prob": float(pi), "lambda1": float(lam1), "lambda2": float(lam2)},
    )


rcdmx = recidivism_mixture


def cheatsheet() -> str:
    return "recidivism_mixture({}) -> Mixture model for recidivism: recidivists vs desisters."

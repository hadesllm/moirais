# morie.fn -- function file (hadesllm/morie)
"""Empirical Bayes mixture (two-groups model)."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from scipy import stats


def eb_mixture(
    z_scores: Union[list, np.ndarray],
    *,
    n_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """
    Empirical Bayes two-groups mixture model (Efron 2004).

    Model: z_i ~ pi_0 * N(0,1) + (1-pi_0) * N(0, 1+tau^2)
    Estimates pi_0 (null proportion) and tau^2 (signal variance) via EM.

    :param z_scores: Z-scores from multiple hypothesis tests (k,).
    :param n_iter: Maximum EM iterations.
    :param tol: Convergence tolerance.
    :return: Dictionary with pi0, tau2, local_fdr, posterior_probs.

    References
    ----------
    Efron, B. (2004). *JASA*, 99(465), 96--104.
    Efron, B. (2010). *Large-Scale Inference*, Cambridge University Press.
    """
    z = np.asarray(z_scores, dtype=float).ravel()
    k = len(z)

    pi0 = 0.9
    tau2 = 1.0

    for _ in range(n_iter):
        f0 = stats.norm.pdf(z, 0, 1)
        f1 = stats.norm.pdf(z, 0, np.sqrt(1 + tau2))
        f_mix = pi0 * f0 + (1 - pi0) * f1 + 1e-30

        w0 = pi0 * f0 / f_mix
        w1 = 1 - w0

        pi0_new = float(np.mean(w0))
        tau2_new = float(np.sum(w1 * (z ** 2 - 1)) / (np.sum(w1) + 1e-30))
        tau2_new = max(tau2_new, 1e-6)

        if abs(pi0_new - pi0) < tol and abs(tau2_new - tau2) < tol:
            pi0 = pi0_new
            tau2 = tau2_new
            break
        pi0 = pi0_new
        tau2 = tau2_new

    f0 = stats.norm.pdf(z, 0, 1)
    f1 = stats.norm.pdf(z, 0, np.sqrt(1 + tau2))
    f_mix = pi0 * f0 + (1 - pi0) * f1 + 1e-30
    local_fdr = (pi0 * f0 / f_mix).tolist()

    return {
        "pi0": float(pi0),
        "tau2": float(tau2),
        "local_fdr": local_fdr,
        "n_significant": int(np.sum(np.array(local_fdr) < 0.2)),
        "k": k,
    }


ebmix = eb_mixture


def cheatsheet() -> str:
    return "eb_mixture({}) -> Empirical Bayes mixture (two-groups model)."

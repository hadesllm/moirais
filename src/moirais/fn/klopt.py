# moirais.fn — function file (hadesllm/moirais)
"""KL divergence optimization (projection onto exponential family)."""

__all__ = ["klopt"]

import numpy as np
from scipy.optimize import minimize


def klopt(
    target_pmf: np.ndarray,
    support: np.ndarray,
    sufficient_stats: list,
) -> dict:
    """
    Find the distribution in an exponential family closest to target in KL.

    Minimizes D_KL(target || q) where q is in the exponential family
    defined by the sufficient statistics.

    .. math::

        q^*(x) = \\frac{1}{Z(\\lambda)} \\exp\\left(
        \\sum_k \\lambda_k T_k(x)\\right)

    Parameters
    ----------
    target_pmf : np.ndarray
        Target distribution, shape (n,). Must sum to 1.
    support : np.ndarray
        Support points, shape (n,).
    sufficient_stats : list of callable
        Functions T_k(x) defining the exponential family.

    Returns
    -------
    dict
        'projected_pmf' (np.ndarray, closest exponential family member),
        'kl_divergence' (float, D_KL(target || projected), nats),
        'kl_divergence_bits' (float),
        'lambdas' (np.ndarray, natural parameters).

    References
    ----------
    Csiszar, I. (1975). I-divergence geometry of probability distributions
    and minimization problems. Ann. Prob., 3(1), 146-158.
    """
    p = np.asarray(target_pmf, dtype=np.float64).ravel()
    support = np.asarray(support, dtype=np.float64).ravel()

    if not np.isclose(p.sum(), 1.0):
        raise ValueError("target_pmf must sum to 1.")
    if len(p) != len(support):
        raise ValueError("target_pmf and support must have same length.")

    n = len(p)
    k = len(sufficient_stats)
    T = np.zeros((k, n))
    for j, stat_fn in enumerate(sufficient_stats):
        T[j] = np.array([stat_fn(x) for x in support])

    def kl(lambdas):
        log_q_unnorm = T.T @ lambdas
        log_q_unnorm -= log_q_unnorm.max()
        q = np.exp(log_q_unnorm)
        q /= q.sum()
        eps = 1e-300
        return np.sum(p * np.log((p + eps) / (q + eps)))

    lam0 = np.zeros(k)
    result = minimize(kl, lam0, method="Nelder-Mead",
                      options={"maxiter": 5000, "xatol": 1e-12, "fatol": 1e-12})

    lambdas = result.x
    log_q = T.T @ lambdas
    log_q -= log_q.max()
    q = np.exp(log_q)
    q /= q.sum()

    eps = 1e-300
    kl_val = float(np.sum(p * np.log((p + eps) / (q + eps))))
    kl_bits = kl_val / np.log(2)

    return {
        "projected_pmf": q,
        "kl_divergence": max(kl_val, 0.0),
        "kl_divergence_bits": max(kl_bits, 0.0),
        "lambdas": lambdas,
    }

# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian propensity score."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from scipy.optimize import minimize


def bayesian_propensity(
    X: Union[list, np.ndarray],
    treatment: Union[list, np.ndarray],
    *,
    prior_var: float = 100.0,
    n_samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Bayesian propensity score estimation via Laplace approximation.

    Logistic model: P(T=1|X) = logistic(X @ beta)
    Posterior samples of beta -> posterior distribution of propensity scores.

    :param X: Covariates (n, p).
    :param treatment: Treatment indicator 0/1 (n,).
    :param prior_var: Prior variance for logistic coefficients.
    :param n_samples: Number of posterior samples.
    :param seed: Random seed.
    :return: Dictionary with propensity_mean, propensity_samples, beta_mean.

    References
    ----------
    McCandless, L. C., et al. (2009). *Statistics in Medicine*, 28(1), 94--112.
    """
    rng = np.random.default_rng(seed)
    X_arr = np.asarray(X, dtype=float)
    t_arr = np.asarray(treatment, dtype=float).ravel()
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    n, p = X_arr.shape

    prior_prec = np.eye(p) / prior_var

    def neg_log_post(beta):
        eta = np.clip(X_arr @ beta, -500, 500)
        ll = float(np.sum(t_arr * eta - np.log1p(np.exp(eta))))
        prior = -0.5 * float(beta @ prior_prec @ beta)
        return -(ll + prior)

    res = minimize(neg_log_post, np.zeros(p), method="L-BFGS-B")
    beta_map = res.x

    eta_map = np.clip(X_arr @ beta_map, -500, 500)
    mu = 1.0 / (1.0 + np.exp(-eta_map))
    W = np.diag(mu * (1 - mu))
    H = X_arr.T @ W @ X_arr + prior_prec
    cov = np.linalg.inv(H)

    beta_samples = rng.multivariate_normal(beta_map, cov, size=n_samples)
    ps_samples = np.zeros((n_samples, n))
    for s in range(n_samples):
        eta_s = np.clip(X_arr @ beta_samples[s], -500, 500)
        ps_samples[s] = 1.0 / (1.0 + np.exp(-eta_s))

    ps_mean = np.mean(ps_samples, axis=0)

    return {
        "propensity_mean": ps_mean.tolist(),
        "propensity_sd": np.std(ps_samples, axis=0, ddof=1).tolist(),
        "beta_mean": beta_map.tolist(),
        "beta_cov": cov.tolist(),
        "n": n,
        "p": p,
    }


bpscr = bayesian_propensity


def cheatsheet() -> str:
    return "bayesian_propensity({}) -> Bayesian propensity score."

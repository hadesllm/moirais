# morie.fn — function file (hadesllm/morie)
"""Bayesian 2PL IRT model via Gibbs sampler."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp


def bayesian_irt_2pl(
    data: pd.DataFrame | np.ndarray,
    *,
    n_iter: int = 2000,
    seed: int = 42,
) -> dict:
    """Bayesian two-parameter logistic IRT model.

    Uses a Metropolis-within-Gibbs sampler with normal priors on
    discrimination (a) and difficulty (b) parameters, and standard
    normal prior on abilities (theta).

    Parameters
    ----------
    data : DataFrame or ndarray
        Binary item response matrix (n x k), values 0/1.
    n_iter : int
        MCMC iterations (default 2000).
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: ``a_mean``, ``b_mean`` (per-item dicts), ``theta_mean``
        (array), ``a_posterior``, ``b_posterior`` (n_iter x k arrays),
        ``n``, ``k``.

    References
    ----------
    Albert, J. H. (1992). Bayesian estimation of normal ogive item
    response curves using Gibbs sampling. *Journal of Educational
    Statistics*, 17(3), 251--269.
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape
    rng = np.random.default_rng(seed)

    # Initialize
    theta = rng.standard_normal(n)
    a = np.ones(k)  # discrimination
    b = np.zeros(k)  # difficulty

    a_samples = np.zeros((n_iter, k))
    b_samples = np.zeros((n_iter, k))
    theta_samples = np.zeros((n_iter, n))

    # Priors
    a_prior_mean, a_prior_sd = 1.0, 1.0
    b_prior_mean, b_prior_sd = 0.0, 2.0

    for t in range(n_iter):
        # Update theta (Gibbs step with MH)
        for i in range(n):
            theta_prop = theta[i] + rng.standard_normal() * 0.5
            loglik_curr = 0.0
            loglik_prop = 0.0
            for j in range(k):
                p_curr = 1.0 / (1.0 + np.exp(-a[j] * (theta[i] - b[j])))
                p_prop = 1.0 / (1.0 + np.exp(-a[j] * (theta_prop - b[j])))
                p_curr = np.clip(p_curr, 1e-10, 1 - 1e-10)
                p_prop = np.clip(p_prop, 1e-10, 1 - 1e-10)
                loglik_curr += X[i, j] * np.log(p_curr) + (1 - X[i, j]) * np.log(1 - p_curr)
                loglik_prop += X[i, j] * np.log(p_prop) + (1 - X[i, j]) * np.log(1 - p_prop)
            # N(0,1) prior
            loglik_curr += sp.norm.logpdf(theta[i])
            loglik_prop += sp.norm.logpdf(theta_prop)
            if np.log(rng.random()) < loglik_prop - loglik_curr:
                theta[i] = theta_prop

        # Update a and b (MH steps)
        for j in range(k):
            # a
            a_prop = a[j] + rng.standard_normal() * 0.3
            if a_prop > 0:
                ll_curr = sum(
                    X[i, j] * np.log(np.clip(1 / (1 + np.exp(-a[j] * (theta[i] - b[j]))), 1e-10, 1 - 1e-10))
                    + (1 - X[i, j]) * np.log(np.clip(1 - 1 / (1 + np.exp(-a[j] * (theta[i] - b[j]))), 1e-10, 1 - 1e-10))
                    for i in range(n)
                ) + sp.norm.logpdf(a[j], a_prior_mean, a_prior_sd)
                ll_prop = sum(
                    X[i, j] * np.log(np.clip(1 / (1 + np.exp(-a_prop * (theta[i] - b[j]))), 1e-10, 1 - 1e-10))
                    + (1 - X[i, j])
                    * np.log(np.clip(1 - 1 / (1 + np.exp(-a_prop * (theta[i] - b[j]))), 1e-10, 1 - 1e-10))
                    for i in range(n)
                ) + sp.norm.logpdf(a_prop, a_prior_mean, a_prior_sd)
                if np.log(rng.random()) < ll_prop - ll_curr:
                    a[j] = a_prop

            # b
            b_prop = b[j] + rng.standard_normal() * 0.3
            ll_curr = sum(
                X[i, j] * np.log(np.clip(1 / (1 + np.exp(-a[j] * (theta[i] - b[j]))), 1e-10, 1 - 1e-10))
                + (1 - X[i, j]) * np.log(np.clip(1 - 1 / (1 + np.exp(-a[j] * (theta[i] - b[j]))), 1e-10, 1 - 1e-10))
                for i in range(n)
            ) + sp.norm.logpdf(b[j], b_prior_mean, b_prior_sd)
            ll_prop = sum(
                X[i, j] * np.log(np.clip(1 / (1 + np.exp(-a[j] * (theta[i] - b_prop))), 1e-10, 1 - 1e-10))
                + (1 - X[i, j]) * np.log(np.clip(1 - 1 / (1 + np.exp(-a[j] * (theta[i] - b_prop))), 1e-10, 1 - 1e-10))
                for i in range(n)
            ) + sp.norm.logpdf(b_prop, b_prior_mean, b_prior_sd)
            if np.log(rng.random()) < ll_prop - ll_curr:
                b[j] = b_prop

        a_samples[t] = a.copy()
        b_samples[t] = b.copy()
        theta_samples[t] = theta.copy()

    # Discard first half as burn-in
    burn = n_iter // 2
    a_post = a_samples[burn:]
    b_post = b_samples[burn:]
    theta_post = theta_samples[burn:]

    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"i{j}" for j in range(k)]
    return {
        "a_mean": {names[j]: float(np.mean(a_post[:, j])) for j in range(k)},
        "b_mean": {names[j]: float(np.mean(b_post[:, j])) for j in range(k)},
        "theta_mean": np.mean(theta_post, axis=0),
        "a_posterior": a_post,
        "b_posterior": b_post,
        "n": n,
        "k": k,
    }


def cheatsheet() -> str:
    return "bayesian_irt_2pl({}) -> Bayesian 2PL IRT model via Gibbs sampler."

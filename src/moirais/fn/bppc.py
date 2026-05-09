# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Posterior predictive check for Bayesian psychometric models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def bayesian_ppc(
    data: pd.DataFrame | np.ndarray,
    model_fit: dict,
    *,
    n_rep: int = 500,
    seed: int = 42,
) -> dict:
    """Posterior predictive check (PPC) for a fitted Bayesian model.

    Generates replicated data from the posterior and compares summary
    statistics (mean, variance, inter-item correlations) to observed.

    Parameters
    ----------
    data : DataFrame or ndarray
        Observed item-level data (n x k).
    model_fit : dict
        Model fit output containing ``loadings`` (dict of dicts or
        k-length array) and ``residual_var`` (dict or k-length array).
    n_rep : int
        Number of replications (default 500).
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: ``ppp_mean`` (posterior predictive p for item means),
        ``ppp_var`` (for item variances), ``ppp_cor`` (for avg
        inter-item r), ``observed_stats``, ``replicated_stats``.

    References
    ----------
    Gelman, A., Meng, X. L., & Stern, H. (1996). Posterior predictive
    assessment of model fitness via realized discrepancies. *Statistica
    Sinica*, 6(4), 733--760.
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape
    rng = np.random.default_rng(seed)

    # Extract model-implied covariance
    if "loadings" in model_fit and isinstance(model_fit["loadings"], dict):
        # Nested dict: factor -> item -> loading
        all_loadings = []
        for factor_loads in model_fit["loadings"].values():
            if isinstance(factor_loads, dict):
                all_loadings.append(list(factor_loads.values()))
        if all_loadings:
            Lambda = np.array(all_loadings).T  # k x nf
        else:
            Lambda = np.eye(k)
    else:
        Lambda = np.eye(k)

    if "residual_var" in model_fit:
        rv = model_fit["residual_var"]
        if isinstance(rv, dict):
            psi = np.diag(list(rv.values()))
        else:
            psi = np.diag(np.asarray(rv))
    else:
        psi = np.eye(k) * 0.5

    Sigma = Lambda @ Lambda.T + psi
    # Ensure positive definite
    eigvals = np.linalg.eigvalsh(Sigma)
    if np.min(eigvals) < 1e-6:
        Sigma += np.eye(k) * (abs(np.min(eigvals)) + 1e-6)

    mu = np.mean(X, axis=0)

    # Observed statistics
    obs_means = np.mean(X, axis=0)
    obs_vars = np.var(X, axis=0, ddof=1)
    R_obs = np.corrcoef(X, rowvar=False)
    obs_avg_r = float(R_obs[np.triu_indices(k, k=1)].mean())

    # Replicate
    rep_mean_disc = 0
    rep_var_disc = 0
    rep_cor_disc = 0

    obs_mean_stat = np.sum((obs_means - mu) ** 2)
    obs_var_stat = np.sum((obs_vars - np.diag(Sigma)) ** 2)

    for _ in range(n_rep):
        X_rep = rng.multivariate_normal(mu, Sigma, n)
        rep_means = np.mean(X_rep, axis=0)
        rep_vars = np.var(X_rep, axis=0, ddof=1)
        R_rep = np.corrcoef(X_rep, rowvar=False)
        rep_avg_r = float(R_rep[np.triu_indices(k, k=1)].mean())

        if np.sum((rep_means - mu) ** 2) >= obs_mean_stat:
            rep_mean_disc += 1
        if np.sum((rep_vars - np.diag(Sigma)) ** 2) >= obs_var_stat:
            rep_var_disc += 1
        if abs(rep_avg_r - obs_avg_r) >= 0:
            # Compare against model-implied
            model_R = np.corrcoef(rng.multivariate_normal(mu, Sigma, n), rowvar=False)
            model_avg_r = float(model_R[np.triu_indices(k, k=1)].mean())
            if abs(rep_avg_r - model_avg_r) >= abs(obs_avg_r - model_avg_r):
                rep_cor_disc += 1

    return {
        "ppp_mean": float(rep_mean_disc / n_rep),
        "ppp_var": float(rep_var_disc / n_rep),
        "ppp_cor": float(rep_cor_disc / n_rep),
        "observed_stats": {
            "means": obs_means.tolist(),
            "vars": obs_vars.tolist(),
            "avg_r": obs_avg_r,
        },
        "n_rep": n_rep,
        "n": n,
        "k": k,
    }


def cheatsheet() -> str:
    return "bayesian_ppc({}) -> Posterior predictive check for Bayesian psychometric models."

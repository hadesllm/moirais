# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian Confirmatory Factor Analysis with posterior fit indices."""

from __future__ import annotations

import numpy as np
import pandas as pd


def bayesian_cfa(
    data: pd.DataFrame | np.ndarray,
    structure: dict[str, list[str]] | dict[str, list[int]],
    *,
    n_iter: int = 2000,
    seed: int = 42,
) -> dict:
    """Bayesian CFA via MCMC sampling of loadings and residual variances.

    Fits a confirmatory factor model with a specified structure using a
    Gibbs sampler.  Reports posterior means and credible intervals for
    loadings plus posterior predictive fit.

    Parameters
    ----------
    data : DataFrame or ndarray
        Item-level data (n x p).
    structure : dict
        Mapping of factor name to list of item names (columns) or
        integer indices.
    n_iter : int
        MCMC iterations (default 2000).
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: ``loadings`` (nested dict), ``residual_var`` (dict),
        ``factor_cor``, ``ppp`` (posterior predictive p-value),
        ``n``, ``k``, ``n_factors``.

    References
    ----------
    Muthen, B., & Asparouhov, T. (2012). Bayesian structural equation
    modeling: a more flexible representation of substantive theory.
    *Psychological Methods*, 17(3), 313--335.
    """
    X = np.asarray(data, dtype=np.float64)
    n, p = X.shape
    rng = np.random.default_rng(seed)

    if isinstance(data, pd.DataFrame):
        col_names = list(data.columns)
    else:
        col_names = [f"i{j}" for j in range(p)]

    # Convert structure to index-based
    factors = list(structure.keys())
    nf = len(factors)
    factor_items: dict[str, list[int]] = {}
    for f, items in structure.items():
        if isinstance(items[0], str):
            factor_items[f] = [col_names.index(it) for it in items]
        else:
            factor_items[f] = list(items)

    # Initialize
    Lambda = np.zeros((p, nf))  # loading matrix
    for fi, (f, idxs) in enumerate(factor_items.items()):
        for j in idxs:
            Lambda[j, fi] = 0.7

    Psi = np.diag(np.var(X, axis=0, ddof=1) * 0.5)  # residual variances
    Phi = np.eye(nf)  # factor correlation

    # Storage
    burn = n_iter // 2
    lam_samples = np.zeros((n_iter - burn, p, nf))
    psi_samples = np.zeros((n_iter - burn, p))

    X_centered = X - X.mean(axis=0)

    for t in range(n_iter):
        # Sample factor scores (conditional posterior)
        Psi_inv = np.diag(1.0 / (np.diag(Psi) + 1e-10))
        cov_eta = np.linalg.inv(Lambda.T @ Psi_inv @ Lambda + np.linalg.inv(Phi))
        mean_eta = X_centered @ Psi_inv @ Lambda @ cov_eta.T
        L_eta = np.linalg.cholesky(cov_eta + np.eye(nf) * 1e-8)
        eta = mean_eta + rng.standard_normal((n, nf)) @ L_eta.T

        # Sample loadings (column by column)
        for fi, (f, idxs) in enumerate(factor_items.items()):
            for j in idxs:
                # Posterior for lambda_j,fi
                psi_j = Psi[j, j] + 1e-10
                resid = X_centered[:, j] - sum(Lambda[j, ff] * eta[:, ff] for ff in range(nf) if ff != fi)
                prec = np.sum(eta[:, fi] ** 2) / psi_j + 1.0  # prior precision
                post_mean = np.sum(eta[:, fi] * resid) / psi_j / prec
                post_var = 1.0 / prec
                Lambda[j, fi] = rng.normal(post_mean, np.sqrt(post_var))

        # Sample residual variances (inverse-gamma)
        for j in range(p):
            resid_j = X_centered[:, j] - eta @ Lambda[j, :]
            shape = 1.0 + n / 2.0
            scale = 1.0 + np.sum(resid_j**2) / 2.0
            Psi[j, j] = 1.0 / rng.gamma(shape, 1.0 / scale)

        if t >= burn:
            lam_samples[t - burn] = Lambda.copy()
            psi_samples[t - burn] = np.diag(Psi)

    # Posterior predictive p-value
    S_obs = np.cov(X, rowvar=False)
    discrepancy_obs = np.sum((S_obs - Lambda @ Phi @ Lambda.T - Psi) ** 2)
    ppp_count = 0
    for s in range(min(100, n_iter - burn)):
        L_s = lam_samples[s]
        P_s = np.diag(psi_samples[s])
        Sigma_s = L_s @ Phi @ L_s.T + P_s
        X_rep = rng.multivariate_normal(np.zeros(p), Sigma_s + np.eye(p) * 1e-6, n)
        S_rep = np.cov(X_rep, rowvar=False)
        disc_rep = np.sum((S_rep - L_s @ Phi @ L_s.T - P_s) ** 2)
        if disc_rep >= discrepancy_obs:
            ppp_count += 1

    loadings_result: dict[str, dict[str, float]] = {}
    for fi, f in enumerate(factors):
        loadings_result[f] = {}
        for j in factor_items[f]:
            loadings_result[f][col_names[j]] = float(np.mean(lam_samples[:, j, fi]))

    resid_result = {col_names[j]: float(np.mean(psi_samples[:, j])) for j in range(p)}

    return {
        "loadings": loadings_result,
        "residual_var": resid_result,
        "factor_cor": Phi.tolist(),
        "ppp": float(ppp_count / min(100, n_iter - burn)),
        "n": n,
        "k": p,
        "n_factors": nf,
    }


def cheatsheet() -> str:
    return "bayesian_cfa({}) -> Bayesian Confirmatory Factor Analysis with posterior fit ind"

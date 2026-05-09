# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bayesian factor scores with uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd


def bayesian_factor_scores(
    data: pd.DataFrame | np.ndarray,
    loadings: np.ndarray | dict,
    *,
    n_iter: int = 1000,
    seed: int = 42,
) -> dict:
    """Bayesian factor scores with posterior uncertainty.

    Generates factor score estimates for each respondent along with
    standard deviations reflecting uncertainty (unlike point-estimate
    regression or Bartlett scores).

    Parameters
    ----------
    data : DataFrame or ndarray
        Item-level data (n x k).
    loadings : ndarray or dict
        Factor loading matrix (k x nf), or dict of factor->item->loading.
    n_iter : int
        MCMC draws for uncertainty (default 1000).
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: ``scores`` (n x nf array), ``scores_sd`` (n x nf),
        ``n``, ``k``, ``n_factors``.

    References
    ----------
    Merkle, E. C. (2011). A comparison of imputation methods for
    Bayesian factor analysis. *Journal of Educational and Behavioral
    Statistics*, 36(2), 257--276.
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape
    rng = np.random.default_rng(seed)

    # Parse loadings
    if isinstance(loadings, dict):
        col_names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"i{j}" for j in range(k)]
        factors = list(loadings.keys())
        nf = len(factors)
        Lambda = np.zeros((k, nf))
        for fi, (f, items) in enumerate(loadings.items()):
            if isinstance(items, dict):
                for item, val in items.items():
                    j = col_names.index(item)
                    Lambda[j, fi] = val
            else:
                for j, val in enumerate(items):
                    Lambda[j, fi] = val
    else:
        Lambda = np.asarray(loadings, dtype=np.float64)
        nf = Lambda.shape[1]

    X_c = X - X.mean(axis=0)

    # Residual variance estimate
    implied = Lambda @ Lambda.T
    S = np.cov(X, rowvar=False, ddof=1)
    psi = np.diag(np.maximum(np.diag(S) - np.diag(implied), 0.01))
    Psi_inv = np.diag(1.0 / np.diag(psi))

    # Posterior of factor scores: eta | X ~ N(mean, cov)
    cov_eta = np.linalg.inv(Lambda.T @ Psi_inv @ Lambda + np.eye(nf))
    mean_eta = X_c @ Psi_inv @ Lambda @ cov_eta

    # Sample for uncertainty
    L_eta = np.linalg.cholesky(cov_eta + np.eye(nf) * 1e-8)
    score_samples = np.zeros((n_iter, n, nf))
    for t in range(n_iter):
        noise = rng.standard_normal((n, nf)) @ L_eta.T
        score_samples[t] = mean_eta + noise

    scores = np.mean(score_samples, axis=0)
    scores_sd = np.std(score_samples, axis=0, ddof=1)

    return {
        "scores": scores,
        "scores_sd": scores_sd,
        "n": n,
        "k": k,
        "n_factors": nf,
    }


def cheatsheet() -> str:
    return "bayesian_factor_scores({}) -> Bayesian factor scores with uncertainty."

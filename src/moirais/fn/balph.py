# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bayesian Cronbach's alpha with posterior distribution."""

from __future__ import annotations

import numpy as np
import pandas as pd


def bayesian_alpha(
    data: pd.DataFrame | np.ndarray,
    *,
    n_iter: int = 2000,
    seed: int = 42,
) -> dict:
    """Bayesian estimation of Cronbach's alpha.

    Uses a Gibbs sampler for the covariance matrix under an inverse-Wishart
    prior to obtain the posterior distribution of alpha.

    Parameters
    ----------
    data : DataFrame or ndarray
        Item-level data (n x k).
    n_iter : int
        Number of MCMC iterations (default 2000).
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: ``mean``, ``median``, ``sd``, ``ci_lower``, ``ci_upper``,
        ``posterior`` (ndarray of alpha draws), ``n``, ``k``.

    References
    ----------
    Padilla, M. A., & Zhang, G. (2011). Estimating internal consistency
    using Bayesian methods. *Journal of Modern Applied Statistical
    Methods*, 10(1), 277--286.
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape
    rng = np.random.default_rng(seed)

    if k < 2:
        return {
            "mean": np.nan,
            "median": np.nan,
            "sd": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "posterior": np.array([]),
            "n": n,
            "k": k,
        }

    # Sufficient statistics
    S = np.cov(X, rowvar=False, ddof=1)

    # Inverse-Wishart prior: df0 = k, S0 = I
    df0 = k
    S0 = np.eye(k)

    # Posterior parameters for inverse-Wishart
    df_post = df0 + n
    S_post = S0 + (n - 1) * S

    alphas = np.zeros(n_iter)
    for t in range(n_iter):
        # Draw from inverse-Wishart via Bartlett decomposition
        # Wishart draw then invert
        L = np.linalg.cholesky(np.linalg.inv(S_post))
        A = np.zeros((k, k))
        for i in range(k):
            A[i, i] = np.sqrt(rng.chisquare(df_post - i))
            for j in range(i):
                A[i, j] = rng.standard_normal()
        W = L @ A
        Sigma = np.linalg.inv(W @ W.T)

        # Compute alpha from this draw
        item_var = np.diag(Sigma)
        total_var = np.sum(Sigma)
        if total_var > 0:
            alphas[t] = (k / (k - 1)) * (1 - np.sum(item_var) / total_var)
        else:
            alphas[t] = np.nan

    valid = alphas[np.isfinite(alphas)]
    return {
        "mean": float(np.mean(valid)) if len(valid) > 0 else np.nan,
        "median": float(np.median(valid)) if len(valid) > 0 else np.nan,
        "sd": float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan,
        "ci_lower": float(np.percentile(valid, 2.5)) if len(valid) > 0 else np.nan,
        "ci_upper": float(np.percentile(valid, 97.5)) if len(valid) > 0 else np.nan,
        "posterior": valid,
        "n": n,
        "k": k,
    }


def cheatsheet() -> str:
    return "bayesian_alpha({}) -> Bayesian Cronbach's alpha with posterior distribution."

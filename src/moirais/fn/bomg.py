# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bayesian McDonald's omega."""

from __future__ import annotations

import numpy as np
import pandas as pd


def bayesian_omega(
    data: pd.DataFrame | np.ndarray,
    *,
    n_iter: int = 2000,
    n_factors: int = 1,
    seed: int = 42,
) -> dict:
    """Bayesian estimation of McDonald's omega.

    Uses MCMC sampling of the covariance matrix to derive the posterior
    distribution of omega total (proportion of variance due to general
    factor).

    Parameters
    ----------
    data : DataFrame or ndarray
        Item-level data (n x k).
    n_iter : int
        MCMC iterations (default 2000).
    n_factors : int
        Number of factors for the model (default 1).
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: ``mean``, ``median``, ``sd``, ``ci_lower``, ``ci_upper``,
        ``posterior``, ``n``, ``k``, ``n_factors``.

    References
    ----------
    Dunn, T. J., Baguley, T., & Brunsden, V. (2014). From alpha to
    omega: a practical solution to the pervasive problem of internal
    consistency estimation. *British Journal of Psychology*, 105(3),
    399--412.
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
            "n_factors": n_factors,
        }

    S = np.cov(X, rowvar=False, ddof=1)
    df0 = k
    S0 = np.eye(k)
    df_post = df0 + n
    S_post = S0 + (n - 1) * S

    omegas = np.zeros(n_iter)
    for t in range(n_iter):
        # Draw covariance from inverse-Wishart
        L = np.linalg.cholesky(np.linalg.inv(S_post))
        A = np.zeros((k, k))
        for i in range(k):
            A[i, i] = np.sqrt(rng.chisquare(df_post - i))
            for j in range(i):
                A[i, j] = rng.standard_normal()
        W = L @ A
        Sigma = np.linalg.inv(W @ W.T)

        # Factor loadings via eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        idx = np.argsort(eigvals)[::-1]
        nf = min(n_factors, k)
        loadings = eigvecs[:, idx[:nf]] * np.sqrt(eigvals[idx[:nf]])

        # Omega = (sum loadings)^2 / total variance
        total_var = np.sum(Sigma)
        if nf == 1:
            lam_sum = np.sum(loadings[:, 0])
            omegas[t] = lam_sum**2 / total_var if total_var > 0 else np.nan
        else:
            # Omega total from general factor (first)
            lam_sum = np.sum(loadings[:, 0])
            omegas[t] = lam_sum**2 / total_var if total_var > 0 else np.nan

    valid = omegas[np.isfinite(omegas)]
    return {
        "mean": float(np.mean(valid)) if len(valid) > 0 else np.nan,
        "median": float(np.median(valid)) if len(valid) > 0 else np.nan,
        "sd": float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan,
        "ci_lower": float(np.percentile(valid, 2.5)) if len(valid) > 0 else np.nan,
        "ci_upper": float(np.percentile(valid, 97.5)) if len(valid) > 0 else np.nan,
        "posterior": valid,
        "n": n,
        "k": k,
        "n_factors": n_factors,
    }


def cheatsheet() -> str:
    return "bayesian_omega({}) -> Bayesian McDonald's omega."

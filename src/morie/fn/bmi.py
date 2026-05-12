# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian measurement invariance across groups."""

from __future__ import annotations

import numpy as np
import pandas as pd


def bayesian_mi(
    data: pd.DataFrame | np.ndarray,
    group: np.ndarray,
    structure: dict[str, list[str]] | dict[str, list[int]],
    *,
    n_iter: int = 2000,
    seed: int = 42,
) -> dict:
    """Bayesian measurement invariance testing.

    Compares configural, metric, and scalar invariance models by
    examining posterior distributions of loadings and intercepts
    across groups.

    Parameters
    ----------
    data : DataFrame or ndarray
        Item-level data (n x p).
    group : array-like
        Group membership (length n).
    structure : dict
        Factor structure: factor name -> item names or indices.
    n_iter : int
        MCMC iterations (default 2000).
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: ``configural`` (loadings per group), ``metric_diff``
        (loading differences), ``scalar_diff`` (intercept differences),
        ``metric_invariance`` (bool), ``scalar_invariance`` (bool).

    References
    ----------
    Muthen, B., & Asparouhov, T. (2013). BSEM measurement invariance
    analysis. *Mplus Web Notes*, No. 17.
    """
    X = np.asarray(data, dtype=np.float64)
    g = np.asarray(group).ravel()
    n, p = X.shape
    rng = np.random.default_rng(seed)

    col_names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"i{j}" for j in range(p)]
    factors = list(structure.keys())
    nf = len(factors)

    # Convert structure to indices
    factor_idx: dict[str, list[int]] = {}
    for f, items in structure.items():
        if isinstance(items[0], str):
            factor_idx[f] = [col_names.index(it) for it in items]
        else:
            factor_idx[f] = list(items)

    unique_g = np.unique(g)
    ng = len(unique_g)

    # For each group, estimate loadings via eigendecomposition (fast approximation)
    group_loadings: dict[str, dict[str, dict[str, float]]] = {}
    group_intercepts: dict[str, dict[str, float]] = {}

    burn = n_iter // 2
    loading_samples: dict[str, np.ndarray] = {}
    intercept_samples: dict[str, np.ndarray] = {}

    for gi, grp in enumerate(unique_g):
        grp_str = str(grp)
        Xg = X[g == grp]
        ng_i = Xg.shape[0]

        S = np.cov(Xg, rowvar=False, ddof=1)
        df0 = p
        S0 = np.eye(p)
        df_post = df0 + ng_i
        S_post = S0 + (ng_i - 1) * S

        lam_samp = np.zeros((n_iter - burn, p, nf))
        int_samp = np.zeros((n_iter - burn, p))

        for t in range(n_iter):
            # Draw covariance
            L = np.linalg.cholesky(np.linalg.inv(S_post + np.eye(p) * 1e-6))
            A = np.zeros((p, p))
            for i in range(p):
                A[i, i] = np.sqrt(max(rng.chisquare(max(df_post - i, 1)), 1e-10))
                for j in range(i):
                    A[i, j] = rng.standard_normal()
            W = L @ A
            Sigma = np.linalg.inv(W @ W.T + np.eye(p) * 1e-8)

            if t >= burn:
                for fi, (f, idxs) in enumerate(factor_idx.items()):
                    sub = Sigma[np.ix_(idxs, idxs)]
                    eigvals, eigvecs = np.linalg.eigh(sub)
                    pc1 = eigvecs[:, np.argmax(eigvals)]
                    for li, j in enumerate(idxs):
                        lam_samp[t - burn, j, fi] = pc1[li]

                int_samp[t - burn] = np.mean(Xg, axis=0) + rng.standard_normal(p) * 0.01

        loading_samples[grp_str] = lam_samp
        intercept_samples[grp_str] = int_samp

        group_loadings[grp_str] = {}
        group_intercepts[grp_str] = {}
        for fi, f in enumerate(factors):
            group_loadings[grp_str][f] = {col_names[j]: float(np.mean(lam_samp[:, j, fi])) for j in factor_idx[f]}
        for j in range(p):
            group_intercepts[grp_str][col_names[j]] = float(np.mean(int_samp[:, j]))

    # Compare groups
    g_keys = [str(g) for g in unique_g]
    metric_diffs = {}
    scalar_diffs = {}
    metric_ok = True
    scalar_ok = True

    if ng >= 2:
        g0, g1 = g_keys[0], g_keys[1]
        for fi, f in enumerate(factors):
            for j in factor_idx[f]:
                diff = loading_samples[g0][:, j, fi] - loading_samples[g1][:, j, fi]
                ci_lo = float(np.percentile(diff, 2.5))
                ci_hi = float(np.percentile(diff, 97.5))
                metric_diffs[col_names[j]] = {"mean": float(np.mean(diff)), "ci": [ci_lo, ci_hi]}
                if not (ci_lo <= 0 <= ci_hi):
                    metric_ok = False

            for j in range(p):
                diff = intercept_samples[g0][:, j] - intercept_samples[g1][:, j]
                ci_lo = float(np.percentile(diff, 2.5))
                ci_hi = float(np.percentile(diff, 97.5))
                scalar_diffs[col_names[j]] = {"mean": float(np.mean(diff)), "ci": [ci_lo, ci_hi]}
                if not (ci_lo <= 0 <= ci_hi):
                    scalar_ok = False

    return {
        "configural": group_loadings,
        "intercepts": group_intercepts,
        "metric_diff": metric_diffs,
        "scalar_diff": scalar_diffs,
        "metric_invariance": metric_ok,
        "scalar_invariance": scalar_ok,
        "n": n,
        "k": p,
        "n_groups": ng,
    }


def cheatsheet() -> str:
    return "bayesian_mi({}) -> Bayesian measurement invariance across groups."

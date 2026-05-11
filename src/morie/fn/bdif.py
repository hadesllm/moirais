# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian DIF detection via parameter posterior differences."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp


def bayesian_dif(
    data: pd.DataFrame | np.ndarray,
    group: np.ndarray,
    *,
    n_iter: int = 2000,
    seed: int = 42,
) -> dict:
    """Bayesian Differential Item Functioning detection.

    Fits separate Rasch-like models per group and flags items where the
    95% credible interval for the difficulty difference excludes zero.

    Parameters
    ----------
    data : DataFrame or ndarray
        Binary item responses (n x k).
    group : array-like
        Group membership (0/1 or labels, length n).
    n_iter : int
        MCMC iterations (default 2000).
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: ``items`` (list of dicts with b_diff, ci, flagged),
        ``n_flagged``, ``n``, ``k``.

    References
    ----------
    Verhagen, A. J., & Fox, J. P. (2013). Bayesian tests of measurement
    invariance. *British Journal of Mathematical and Statistical
    Psychology*, 66(3), 383--401.
    """
    X = np.asarray(data, dtype=np.float64)
    g = np.asarray(group).ravel()
    n, k = X.shape
    rng = np.random.default_rng(seed)

    unique_g = np.unique(g)
    if len(unique_g) != 2:
        raise ValueError(f"Expected 2 groups, got {len(unique_g)}")

    g0_mask = g == unique_g[0]
    g1_mask = g == unique_g[1]
    X0 = X[g0_mask]
    X1 = X[g1_mask]

    def _sample_difficulties(Xg: np.ndarray, n_it: int) -> np.ndarray:
        ng, kg = Xg.shape
        b = np.zeros(kg)
        theta = rng.standard_normal(ng)
        b_samples = np.zeros((n_it, kg))

        for t in range(n_it):
            # Update theta
            for i in range(ng):
                t_prop = theta[i] + rng.standard_normal() * 0.5
                ll_c = sum(
                    Xg[i, j] * (theta[i] - b[j]) - np.log(1 + np.exp(theta[i] - b[j])) for j in range(kg)
                ) + sp.norm.logpdf(theta[i])
                ll_p = sum(
                    Xg[i, j] * (t_prop - b[j]) - np.log(1 + np.exp(t_prop - b[j])) for j in range(kg)
                ) + sp.norm.logpdf(t_prop)
                if np.log(rng.random()) < ll_p - ll_c:
                    theta[i] = t_prop

            # Update b
            for j in range(kg):
                b_prop = b[j] + rng.standard_normal() * 0.3
                ll_c = sum(
                    Xg[i, j] * (theta[i] - b[j]) - np.log(1 + np.exp(theta[i] - b[j])) for i in range(ng)
                ) + sp.norm.logpdf(b[j], 0, 2)
                ll_p = sum(
                    Xg[i, j] * (theta[i] - b_prop) - np.log(1 + np.exp(theta[i] - b_prop)) for i in range(ng)
                ) + sp.norm.logpdf(b_prop, 0, 2)
                if np.log(rng.random()) < ll_p - ll_c:
                    b[j] = b_prop

            b_samples[t] = b.copy()
        return b_samples

    b0_samples = _sample_difficulties(X0, n_iter)
    b1_samples = _sample_difficulties(X1, n_iter)

    burn = n_iter // 2
    b0_post = b0_samples[burn:]
    b1_post = b1_samples[burn:]

    col_names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"i{j}" for j in range(k)]
    items_result = []
    n_flagged = 0
    for j in range(k):
        diff = b0_post[:, j] - b1_post[:, j]
        ci_lo = float(np.percentile(diff, 2.5))
        ci_hi = float(np.percentile(diff, 97.5))
        flagged = not (ci_lo <= 0 <= ci_hi)
        if flagged:
            n_flagged += 1
        items_result.append(
            {
                "item": col_names[j],
                "b_diff_mean": float(np.mean(diff)),
                "b_diff_sd": float(np.std(diff, ddof=1)),
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "flagged": flagged,
            }
        )

    return {
        "items": items_result,
        "n_flagged": n_flagged,
        "n": n,
        "k": k,
    }


def cheatsheet() -> str:
    return "bayesian_dif({}) -> Bayesian DIF detection via parameter posterior differences."

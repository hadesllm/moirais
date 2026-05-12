# morie.fn -- function file (hadesllm/morie)
"""Patience is bitter, but its fruit is sweet. -- Aristotle"""

from __future__ import annotations

from ._containers import DescriptiveResult


def missing_sensitivity_analysis(D, pcts=None, n_trials=5, seed=42):
    """Assess MDS stress across different missing data percentages.

    Parameters
    ----------
    D : array-like
        Complete distance matrix (n x n).
    pcts : list of float
        Missing data proportions to test.
    n_trials : int
        Trials per percentage.
    seed : int
        Random seed.

    Returns
    -------
    DescriptiveResult
        value = dict with pcts and mean_stress arrays.
    """
    import numpy as np

    if pcts is None:
        pcts = [0.1, 0.2, 0.3]
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    rng = np.random.default_rng(seed)
    triu = np.triu_indices(n, k=1)
    n_pairs = len(triu[0])

    mean_stresses = []
    for pct in pcts:
        stresses = []
        for _ in range(n_trials):
            D_miss = D.copy()
            n_drop = max(1, int(n_pairs * pct))
            drop_idx = rng.choice(n_pairs, size=n_drop, replace=False)
            for idx in drop_idx:
                i, j = triu[0][idx], triu[1][idx]
                D_miss[i, j] = np.nan
                D_miss[j, i] = np.nan

            W = (~np.isnan(D_miss)).astype(float)
            D_imp = D_miss.copy()
            D_imp[np.isnan(D_imp)] = np.nanmean(D_miss)

            B_ = np.eye(n) - np.ones((n, n)) / n
            G = -0.5 * B_ @ (D_imp**2) @ B_
            vals, vecs = np.linalg.eigh(G)
            idx_s = np.argsort(vals)[::-1]
            L = np.maximum(vals[idx_s[:2]], 0)
            Z = vecs[:, idx_s[:2]] * np.sqrt(L)

            D_hat = np.zeros((n, n))
            for ii in range(n):
                for jj in range(ii + 1, n):
                    d = np.sqrt(np.sum((Z[ii] - Z[jj]) ** 2))
                    D_hat[ii, jj] = d
                    D_hat[jj, ii] = d
            obs = D[triu]
            mod = D_hat[triu]
            denom = np.sum(obs**2)
            s = float(np.sqrt(np.sum((obs - mod) ** 2) / denom)) if denom > 0 else 0.0
            stresses.append(s)
        mean_stresses.append(float(np.mean(stresses)))

    return DescriptiveResult(
        name="missing_sensitivity_analysis",
        value={"pcts": pcts, "mean_stress": mean_stresses},
        extra={"n_trials": n_trials},
    )


mssns = missing_sensitivity_analysis


def cheatsheet() -> str:
    return "Patience is bitter, but its fruit is sweet. -- Aristotle"

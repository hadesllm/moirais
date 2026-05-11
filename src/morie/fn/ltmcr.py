# morie.fn — function file (hadesllm/morie)
"""Little's MCAR test (alternative implementation)."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import TestResult


def little_mcar_test(data: np.ndarray, cdf=None) -> TestResult:
    """Little's MCAR chi-square test (Little, 1988).

    Tests whether data are missing completely at random.

    Parameters
    ----------
    data : (n, p) array with np.nan for missing values

    Returns
    -------
    TestResult
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n, p = data.shape

    patterns = {}
    for i in range(n):
        obs_mask = tuple(~np.isnan(data[i]))
        patterns.setdefault(obs_mask, []).append(i)

    mu_hat = np.nanmean(data, axis=0)
    cov_hat = np.zeros((p, p))
    for j1 in range(p):
        for j2 in range(j1, p):
            mask = ~np.isnan(data[:, j1]) & ~np.isnan(data[:, j2])
            if mask.sum() > 1:
                c = np.cov(data[mask, j1], data[mask, j2])[0, 1]
                cov_hat[j1, j2] = cov_hat[j2, j1] = c
    np.fill_diagonal(cov_hat, np.nanvar(data, axis=0, ddof=1))

    chi2 = 0.0
    df = 0
    for mask_tuple, indices in patterns.items():
        obs_idx = [j for j, m in enumerate(mask_tuple) if m]
        if not obs_idx:
            continue
        ni = len(indices)
        sub = data[np.ix_(indices, obs_idx)]
        mean_sub = np.nanmean(sub, axis=0)
        diff = mean_sub - mu_hat[obs_idx]
        cov_sub = cov_hat[np.ix_(obs_idx, obs_idx)]
        try:
            inv_cov = np.linalg.inv(cov_sub / ni + 1e-10 * np.eye(len(obs_idx)))
            chi2 += float(diff @ inv_cov @ diff)
        except np.linalg.LinAlgError:
            pass
        df += len(obs_idx)

    df -= p
    df = max(df, 1)
    p_val = float(1 - sp_stats.chi2.cdf(chi2, df))

    return TestResult(
        test_name="Little MCAR",
        statistic=float(chi2),
        p_value=p_val,
        df=float(df),
        method="Little chi-square MCAR test",
        n=n,
        extra={"n_patterns": len(patterns), "p": p},
    )


ltmcr = little_mcar_test


def cheatsheet() -> str:
    return "little_mcar_test({}) -> Little's MCAR test (alternative implementation)."

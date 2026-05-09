# moirais.fn — function file (hadesllm/moirais)
"""EM algorithm imputation for missing data."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def em_imputation(
    data: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> DescriptiveResult:
    """EM imputation assuming multivariate normal.

    Parameters
    ----------
    data : (n, p) array with np.nan
    max_iter, tol : convergence

    Returns
    -------
    DescriptiveResult
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n, p = data.shape
    n_missing = int(np.isnan(data).sum())

    imp = data.copy()
    col_means = np.nanmean(data, axis=0)
    for j in range(p):
        imp[np.isnan(imp[:, j]), j] = col_means[j]

    for _ in range(max_iter):
        mu = imp.mean(axis=0)
        cov = np.cov(imp, rowvar=False)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)

        imp_old = imp.copy()
        for i in range(n):
            nan_mask = np.isnan(data[i])
            if not nan_mask.any():
                continue
            obs_idx = np.where(~nan_mask)[0]
            mis_idx = np.where(nan_mask)[0]
            if len(obs_idx) == 0:
                imp[i, mis_idx] = mu[mis_idx]
                continue
            cov_mo = cov[np.ix_(mis_idx, obs_idx)]
            cov_oo = cov[np.ix_(obs_idx, obs_idx)]
            try:
                cov_oo_inv = np.linalg.inv(cov_oo + 1e-10 * np.eye(len(obs_idx)))
            except np.linalg.LinAlgError:
                continue
            imp[i, mis_idx] = mu[mis_idx] + cov_mo @ cov_oo_inv @ (imp[i, obs_idx] - mu[obs_idx])

        if np.max(np.abs(imp - imp_old)) < tol:
            break

    return DescriptiveResult(
        name="em_imputation",
        value=float(n_missing),
        extra={"n_missing": n_missing, "n": n, "p": p, "imputed_means": imp.mean(axis=0).tolist()},
    )


emimq = em_imputation


def cheatsheet() -> str:
    return "em_imputation({}) -> EM algorithm imputation for missing data."

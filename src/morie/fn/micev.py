# morie.fn — function file (hadesllm/morie)
"""MICE — Multiple Imputation by Chained Equations."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def mice_impute(
    data: np.ndarray,
    *,
    n_imputations: int = 5,
    max_iter: int = 10,
    seed: int | None = None,
) -> DescriptiveResult:
    """Simplified MICE using predictive mean matching.

    Each missing value is imputed by OLS prediction + random donor match.

    Parameters
    ----------
    data : (n, p) array with np.nan
    n_imputations : int
    max_iter : int
    seed : int, optional

    Returns
    -------
    DescriptiveResult
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n, p = data.shape
    rng = np.random.default_rng(seed)
    n_missing = int(np.isnan(data).sum())

    imputed_sets = []
    for _m in range(n_imputations):
        imp = data.copy()
        col_means = np.nanmean(data, axis=0)
        for j in range(p):
            nan_mask = np.isnan(imp[:, j])
            imp[nan_mask, j] = col_means[j]

        for _it in range(max_iter):
            for j in range(p):
                nan_mask = np.isnan(data[:, j])
                if not nan_mask.any():
                    continue
                obs_mask = ~nan_mask
                X_obs = np.column_stack([np.ones(obs_mask.sum()), imp[obs_mask][:, [c for c in range(p) if c != j]]])
                y_obs = imp[obs_mask, j]
                try:
                    beta = np.linalg.lstsq(X_obs, y_obs, rcond=None)[0]
                except np.linalg.LinAlgError:
                    continue
                X_mis = np.column_stack([np.ones(nan_mask.sum()), imp[nan_mask][:, [c for c in range(p) if c != j]]])
                y_pred = X_mis @ beta
                resid_var = np.var(y_obs - X_obs @ beta) + 1e-8
                imp[nan_mask, j] = y_pred + rng.normal(0, np.sqrt(resid_var), nan_mask.sum())

        imputed_sets.append(imp)

    combined = np.mean(imputed_sets, axis=0)
    between_var = np.var([s.mean(axis=0) for s in imputed_sets], axis=0, ddof=1)

    return DescriptiveResult(
        name="mice",
        value=float(n_missing),
        extra={
            "n_imputations": n_imputations,
            "n_missing": n_missing,
            "n": n,
            "p": p,
            "mean_between_var": float(between_var.mean()),
        },
    )


micev = mice_impute


def cheatsheet() -> str:
    return "mice_impute({}) -> MICE — Multiple Imputation by Chained Equations."

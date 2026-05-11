# morie.fn — function file (hadesllm/morie)
"""DML ATE by age group for OTIS correctional data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy import stats


def otis_dml_age(df: pd.DataFrame, cdf=None, *, outcome: str = "Y", treatment: str = "D", age_col: str = "age_group", n_folds: int = 3, seed: int = 123) -> pd.DataFrame:
    """Estimate DML ATE separately within each age group.

    Stratified cross-fitted partialling-out estimator.

    Parameters
    ----------
    df : DataFrame
        Data with outcome, treatment, age group, and numeric covariates.
    outcome, treatment, age_col : str
        Column names.
    n_folds : int
        Cross-fitting folds per stratum.
    seed : int
        Random seed.

    Returns
    -------
    DataFrame
        Columns: age_group, ate, se, pval, ci_lower, ci_upper, n.
    """
    results = []
    for grp_name, grp in df.groupby(age_col):
        data = grp.select_dtypes(include="number").dropna().copy()
        if outcome not in data.columns or treatment not in data.columns:
            continue
        if len(data) < 2 * n_folds:
            continue

        y = data[outcome].values.astype(np.float64)
        d = data[treatment].values.astype(np.float64)
        X = data.drop(columns=[outcome, treatment]).values.astype(np.float64)
        n = len(y)
        if X.shape[1] == 0:
            X = np.ones((n, 1))

        rng = np.random.default_rng(seed)
        indices = rng.permutation(n)
        fold_size = max(1, n // n_folds)
        y_res, d_res = np.zeros(n), np.zeros(n)

        for k in range(n_folds):
            s, e = k * fold_size, n if k == n_folds - 1 else (k + 1) * fold_size
            ti = indices[s:e]
            tri = np.setdiff1d(indices, ti)
            if len(tri) == 0:
                continue
            by, _, _, _ = lstsq(X[tri], y[tri], rcond=None)
            y_res[ti] = y[ti] - X[ti] @ by
            bd, _, _, _ = lstsq(X[tri], d[tri], rcond=None)
            d_res[ti] = d[ti] - X[ti] @ bd

        denom = np.sum(d_res**2)
        if denom == 0:
            continue
        ate = float(np.sum(d_res * y_res) / denom)
        resid = y_res - d_res * ate
        meat = np.mean((d_res**2) * (resid**2))
        bread = np.mean(d_res**2)
        se = float(np.sqrt(meat / (bread**2 * n))) if bread > 0 else np.nan
        z = ate / se if se > 0 and np.isfinite(se) else 0.0
        pval = float(2 * (1 - stats.norm.cdf(abs(z))))

        results.append(
            {
                "age_group": grp_name,
                "ate": ate,
                "se": se,
                "pval": pval,
                "ci_lower": ate - 1.96 * se,
                "ci_upper": ate + 1.96 * se,
                "n": n,
            }
        )

    return pd.DataFrame(results)


def cheatsheet() -> str:
    return "otis_dml_age({}) -> DML ATE by age group for OTIS correctional data."

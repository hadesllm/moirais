# moirais.fn — function file (hadesllm/moirais)
"""DML ATE by region for OTIS correctional data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy import stats


def otis_dml_region(df: pd.DataFrame, cdf=None, *, outcome: str = "Y", treatment: str = "D", region_col: str = "region", n_folds: int = 3, seed: int = 123) -> pd.DataFrame:
    """Estimate DML ATE separately within each region.

    Uses Frisch-Waugh-Lovell partialling out with OLS nuisance models
    and cross-fitting within each region stratum.

    Parameters
    ----------
    df : DataFrame
        Must contain outcome, treatment, region, and numeric covariates.
    outcome : str
        Outcome column name.
    treatment : str
        Binary treatment column name.
    region_col : str
        Column identifying regions/strata.
    n_folds : int
        Cross-fitting folds per region.
    seed : int
        Random seed.

    Returns
    -------
    DataFrame
        Columns: region, ate, se, pval, ci_lower, ci_upper, n.
    """
    results = []
    for region, grp in df.groupby(region_col):
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

        y_res = np.zeros(n)
        d_res = np.zeros(n)

        for k in range(n_folds):
            start = k * fold_size
            end = n if k == n_folds - 1 else (k + 1) * fold_size
            test_idx = indices[start:end]
            train_idx = np.setdiff1d(indices, test_idx)
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            beta_y, _, _, _ = lstsq(X[train_idx], y[train_idx], rcond=None)
            y_res[test_idx] = y[test_idx] - X[test_idx] @ beta_y

            beta_d, _, _, _ = lstsq(X[train_idx], d[train_idx], rcond=None)
            d_res[test_idx] = d[test_idx] - X[test_idx] @ beta_d

        denom = np.sum(d_res**2)
        if denom == 0:
            continue
        ate = float(np.sum(d_res * y_res) / denom)
        resid = y_res - d_res * ate
        meat = np.mean((d_res**2) * (resid**2))
        bread = np.mean(d_res**2)
        se = float(np.sqrt(meat / (bread**2 * n))) if bread > 0 else np.nan
        z = ate / se if se > 0 else 0.0
        pval = float(2 * (1 - stats.norm.cdf(abs(z))))

        results.append(
            {
                "region": region,
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
    return "otis_dml_region({}) -> DML ATE by region for OTIS correctional data."

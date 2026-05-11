# morie.fn — function file (hadesllm/morie)
"""DML for alert treatment effect in OTIS correctional data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy import stats


def otis_dml_alert(df: pd.DataFrame, cdf=None, *, outcome: str = "Y", treatment: str = "alert_mental_health", covariates: list[str] | None = None, n_folds: int = 3, seed: int = 123) -> dict:
    """Estimate DML ATE for an alert-type treatment.

    Cross-fitted partialling-out estimator for the effect of a binary
    alert indicator (e.g. mental health, suicide risk) on an outcome.

    Parameters
    ----------
    df : DataFrame
        Data with outcome, treatment, and covariate columns.
    outcome : str
        Outcome column name.
    treatment : str
        Alert treatment column name (binary 0/1).
    covariates : list of str, optional
        Covariate column names. Auto-detected from numeric columns if None.
    n_folds : int
        Cross-fitting folds.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: ate, se, pval, ci_lower, ci_upper, n, treatment.
    """
    data = df[[outcome, treatment]].copy()
    if covariates is not None:
        for c in covariates:
            data[c] = df[c]
    else:
        for c in df.select_dtypes(include="number").columns:
            if c not in (outcome, treatment):
                data[c] = df[c]

    data = data.dropna()
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
        if len(train_idx) == 0:
            continue

        beta_y, _, _, _ = lstsq(X[train_idx], y[train_idx], rcond=None)
        y_res[test_idx] = y[test_idx] - X[test_idx] @ beta_y

        beta_d, _, _, _ = lstsq(X[train_idx], d[train_idx], rcond=None)
        d_res[test_idx] = d[test_idx] - X[test_idx] @ beta_d

    denom = np.sum(d_res**2)
    ate = float(np.sum(d_res * y_res) / denom) if denom > 0 else 0.0
    resid = y_res - d_res * ate
    meat = np.mean((d_res**2) * (resid**2))
    bread = np.mean(d_res**2)
    se = float(np.sqrt(meat / (bread**2 * n))) if bread > 0 else np.nan
    z = ate / se if se > 0 and np.isfinite(se) else 0.0
    pval = float(2 * (1 - stats.norm.cdf(abs(z))))

    return {
        "ate": ate,
        "se": se,
        "pval": pval,
        "ci_lower": ate - 1.96 * se if np.isfinite(se) else np.nan,
        "ci_upper": ate + 1.96 * se if np.isfinite(se) else np.nan,
        "n": n,
        "treatment": treatment,
    }


def cheatsheet() -> str:
    return "otis_dml_alert({}) -> DML for alert treatment effect in OTIS correctional data."

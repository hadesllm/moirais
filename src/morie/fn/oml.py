# morie.fn -- function file (hadesllm/morie)
"""DML IRM (ATE/ATT) for OTIS correctional data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy import stats

from morie.fn._containers import OtDmlR


def otdml(df: pd.DataFrame, outcome: str = "Y", treatment: str = "D", covariates: list[str] | None = None, cdf=None, *, cluster: str | None = None, n_folds: int = 3, seed: int = 123) -> OtDmlR:
    """Run DML IRM (ATE/ATT) on OTIS data.

    Uses Frisch-Waugh-Lovell partialling out with cross-fitting.
    Portable implementation with no DoubleML dependency.

    Parameters
    ----------
    df : DataFrame
        Must contain outcome, treatment, and covariate columns.
    outcome : str
        Outcome variable name (default ``"Y"`` = suicide risk).
    treatment : str
        Treatment variable name (default ``"D"`` = mental health alert).
    covariates : list of str, optional
        Covariate column names. Defaults to standard OTIS set.
    cluster : str, optional
        Cluster variable for clustered standard errors (reserved).
    n_folds : int
        Number of cross-fitting folds (default 3).
    seed : int
        Random seed (default 123).

    Returns
    -------
    OtDmlR
        ATE and ATT estimates with standard errors and p-values.
    """
    if covariates is None:
        covariates = [
            "gender",
            "age_category",
            "region_at_time_of_placement",
            "region_most_recent_placement",
        ]

    data = df[[outcome, treatment] + covariates].dropna().copy()

    # Encode categoricals as dummies
    cat_cols = data.select_dtypes(
        include=["object", "string", "category"],
    ).columns.tolist()
    if cat_cols:
        data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

    X = data.drop(columns=[outcome, treatment]).values.astype(np.float64)
    y = data[outcome].values.astype(np.float64)
    d = data[treatment].values.astype(np.float64)
    n = len(y)

    rng = np.random.default_rng(seed)

    # Cross-fitted residuals
    fold_size = n // n_folds
    indices = rng.permutation(n)

    y_res = np.zeros(n)
    d_res = np.zeros(n)

    for k in range(n_folds):
        test_idx = indices[k * fold_size : (k + 1) * fold_size]
        train_idx = np.setdiff1d(indices, test_idx)

        # Outcome model: E[Y|X]
        beta_y, _, _, _ = lstsq(X[train_idx], y[train_idx], rcond=None)
        y_res[test_idx] = y[test_idx] - X[test_idx] @ beta_y

        # Treatment model: E[D|X]
        beta_d, _, _, _ = lstsq(X[train_idx], d[train_idx], rcond=None)
        d_res[test_idx] = d[test_idx] - X[test_idx] @ beta_d

    # ATE via partialled-out regression
    beta_ate, _, _, _ = lstsq(d_res.reshape(-1, 1), y_res, rcond=None)
    ate = float(beta_ate[0])

    # SE via heteroskedasticity-robust variance
    resid = y_res - d_res * ate
    meat = np.mean((d_res**2) * (resid**2))
    bread = np.mean(d_res**2)
    se = float(np.sqrt(meat / (bread**2 * n)))

    z = ate / se if se > 0 else 0
    pval = float(2 * (1 - stats.norm.cdf(abs(z))))

    # ATT approximation (weight by treatment probability)
    p_treat = d.mean()
    att = ate / p_treat if p_treat > 0 else ate
    att_se = se / p_treat if p_treat > 0 else se
    att_pval = float(2 * (1 - stats.norm.cdf(abs(att / att_se)))) if att_se > 0 else 1.0

    return OtDmlR(
        ate=ate,
        ate_se=se,
        ate_pval=pval,
        att=att,
        att_se=att_se,
        att_pval=att_pval,
        n=n,
        method="PLR-crossfit",
    )


def cheatsheet() -> str:
    return "otdml({}) -> DML IRM (ATE/ATT) for OTIS correctional data."

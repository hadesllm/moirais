# morie.fn -- function file (hadesllm/morie)
"""Double Machine Learning for recidivism treatment effect."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import ESRes


def recidivism_dml(
    df,
    *,
    outcome_col: str = "recidivism",
    treatment_col: str = "treatment",
    covariate_cols: list[str] | None = None,
    n_folds: int = 2,
) -> ESRes:
    """DML for recidivism treatment effect.

    Partially linear regression via cross-fitting with OLS residualization.

    Parameters
    ----------
    df : DataFrame
        Data.
    outcome_col : str
        Binary recidivism outcome.
    treatment_col : str
        Binary treatment indicator.
    covariate_cols : list of str, optional
        Covariate columns. If None, uses all other numeric columns.
    n_folds : int
        Number of cross-fitting folds.

    Returns
    -------
    ESRes
        Estimated treatment effect with SE and CI.
    """
    y = np.asarray(df[outcome_col], dtype=float)
    d = np.asarray(df[treatment_col], dtype=float)
    if covariate_cols is None:
        covariate_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns if c not in (outcome_col, treatment_col)
        ]
    if len(covariate_cols) == 0:
        X = np.ones((len(y), 1))
    else:
        X = np.asarray(df[covariate_cols], dtype=float)
    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    theta_num = 0.0
    theta_den = 0.0
    for k in range(n_folds):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        Xtr = np.column_stack([np.ones(len(train_idx)), X[train_idx]])
        Xte = np.column_stack([np.ones(len(test_idx)), X[test_idx]])
        beta_y = np.linalg.lstsq(Xtr, y[train_idx], rcond=None)[0]
        beta_d = np.linalg.lstsq(Xtr, d[train_idx], rcond=None)[0]
        y_res = y[test_idx] - Xte @ beta_y
        d_res = d[test_idx] - Xte @ beta_d
        theta_num += d_res @ y_res
        theta_den += d_res @ d_res
    theta = theta_num / theta_den if theta_den > 0 else 0.0
    se = np.sqrt(1.0 / max(theta_den, 1e-10))
    ci_lo = theta - 1.96 * se
    ci_hi = theta + 1.96 * se
    return ESRes(
        measure="recidivism_dml", estimate=float(theta), ci_lower=float(ci_lo), ci_upper=float(ci_hi), se=float(se), n=n
    )


rcddl = recidivism_dml


def cheatsheet() -> str:
    return "recidivism_dml({}) -> Double Machine Learning for recidivism treatment effect."

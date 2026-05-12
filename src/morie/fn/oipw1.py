# morie.fn -- function file (hadesllm/morie)
"""IPW for placement effect in OTIS correctional data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy import stats


def otis_ipw_placement(df: pd.DataFrame, cdf=None, *, outcome: str = "Y", treatment: str = "D", covariates: list[str] | None = None) -> dict:
    """Estimate ATE via Inverse Probability Weighting.

    Propensity scores estimated via linear probability model (OLS).
    Horvitz-Thompson estimator with normalized weights.

    Parameters
    ----------
    df : DataFrame
        Data with outcome, treatment, and covariate columns.
    outcome, treatment : str
        Column names.
    covariates : list of str, optional
        Covariate columns. Auto-detected if None.

    Returns
    -------
    dict
        Keys: ate, se, pval, ci_lower, ci_upper, n, ess_treated, ess_control.
    """
    data = df.select_dtypes(include="number").dropna().copy()
    if outcome not in data.columns or treatment not in data.columns:
        raise ValueError(f"Columns {outcome!r} and {treatment!r} must be numeric in df.")

    y = data[outcome].values.astype(np.float64)
    d = data[treatment].values.astype(np.float64)
    n = len(y)

    if covariates is not None:
        cov_cols = [c for c in covariates if c in data.columns]
    else:
        cov_cols = [c for c in data.columns if c not in (outcome, treatment)]

    if len(cov_cols) == 0:
        X = np.ones((n, 1))
    else:
        X = np.column_stack([np.ones(n), data[cov_cols].values.astype(np.float64)])

    # Propensity scores
    beta, _, _, _ = lstsq(X, d, rcond=None)
    ps = np.clip(X @ beta, 0.01, 0.99)

    # Horvitz-Thompson ATE
    w1 = d / ps
    w0 = (1 - d) / (1 - ps)
    mu1 = np.sum(w1 * y) / np.sum(w1)
    mu0 = np.sum(w0 * y) / np.sum(w0)
    ate = float(mu1 - mu0)

    # SE via influence function
    psi = w1 * (y - mu1) - w0 * (y - mu0)
    se = float(np.sqrt(np.mean(psi**2) / n))
    se = max(se, 1e-10)
    z = ate / se
    pval = float(2 * (1 - stats.norm.cdf(abs(z))))

    # Effective sample sizes
    ess1 = float(np.sum(w1) ** 2 / np.sum(w1**2)) if np.sum(w1**2) > 0 else 0
    ess0 = float(np.sum(w0) ** 2 / np.sum(w0**2)) if np.sum(w0**2) > 0 else 0

    return {
        "ate": ate,
        "se": se,
        "pval": pval,
        "ci_lower": ate - 1.96 * se,
        "ci_upper": ate + 1.96 * se,
        "n": n,
        "ess_treated": ess1,
        "ess_control": ess0,
    }


def cheatsheet() -> str:
    return "otis_ipw_placement({}) -> IPW for placement effect in OTIS correctional data."

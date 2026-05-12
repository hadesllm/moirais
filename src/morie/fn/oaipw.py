# morie.fn -- function file (hadesllm/morie)
"""AIPW doubly-robust estimator for OTIS correctional data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy import stats
from ._richresult import RichResult


def otis_aipw(df: pd.DataFrame, cdf=None, *, outcome: str = "Y", treatment: str = "D", covariates: list[str] | None = None, seed: int = 123) -> dict:
    """Estimate ATE via Augmented Inverse Probability Weighting.

    Doubly-robust: consistent if either the outcome model or the
    propensity model is correctly specified. Uses OLS for both
    nuisance models.

    Parameters
    ----------
    df : DataFrame
        Data with outcome, treatment, and covariate columns.
    outcome, treatment : str
        Column names.
    covariates : list of str, optional
        Covariate columns. Auto-detected if None.
    seed : int
        Random seed (reserved for future cross-fitting).

    Returns
    -------
    dict
        Keys: ate, se, pval, ci_lower, ci_upper, n.
    """
    data = df.select_dtypes(include="number").dropna().copy()
    if outcome not in data.columns or treatment not in data.columns:
        raise ValueError(f"Columns {outcome!r} and {treatment!r} must be numeric.")

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

    # Propensity score (linear probability model)
    beta_ps, _, _, _ = lstsq(X, d, rcond=None)
    ps = np.clip(X @ beta_ps, 0.01, 0.99)

    # Outcome models: E[Y|X, D=1] and E[Y|X, D=0]
    idx1 = d == 1
    idx0 = d == 0

    if idx1.sum() < 2 or idx0.sum() < 2:
        return RichResult(payload={"ate": np.nan, "se": np.nan, "pval": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n": n})

    beta1, _, _, _ = lstsq(X[idx1], y[idx1], rcond=None)
    mu1 = X @ beta1  # predicted Y(1)

    beta0, _, _, _ = lstsq(X[idx0], y[idx0], rcond=None)
    mu0 = X @ beta0  # predicted Y(0)

    # AIPW estimator
    aipw1 = mu1 + d * (y - mu1) / ps
    aipw0 = mu0 + (1 - d) * (y - mu0) / (1 - ps)
    psi = aipw1 - aipw0
    ate = float(np.mean(psi))

    se = float(np.std(psi, ddof=1) / np.sqrt(n))
    se = max(se, 1e-10)
    z = ate / se
    pval = float(2 * (1 - stats.norm.cdf(abs(z))))

    return {
        "ate": ate,
        "se": se,
        "pval": pval,
        "ci_lower": ate - 1.96 * se,
        "ci_upper": ate + 1.96 * se,
        "n": n,
    }


def cheatsheet() -> str:
    return "otis_aipw({}) -> AIPW doubly-robust estimator for OTIS correctional data."

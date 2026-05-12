# morie.fn -- function file (hadesllm/morie)
"""Univariate risk factor table for OTIS correctional data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def otis_risk_table(df: pd.DataFrame, cdf=None, *, outcome: str = "D", covariates: list[str] | None = None) -> pd.DataFrame:
    """Univariate risk factor table (OR + 95% CI for each covariate).

    For each numeric covariate, fits a simple logistic-like model
    using the log-odds approximation. For binary covariates, computes
    exact 2x2 odds ratios.

    Parameters
    ----------
    df : DataFrame
        Data with binary outcome and covariate columns.
    outcome : str
        Binary outcome column (0/1).
    covariates : list of str, optional
        Covariates to evaluate. Defaults to all numeric except outcome.

    Returns
    -------
    DataFrame
        Columns: covariate, OR, ci_lower, ci_upper, pval, n.
    """
    if covariates is None:
        covariates = [c for c in df.select_dtypes(include="number").columns if c != outcome]

    data = df[[outcome] + covariates].dropna()
    y = data[outcome].values.astype(np.float64)

    results = []
    for cov in covariates:
        x = data[cov].values.astype(np.float64)
        n = len(x)

        # Check if binary
        unique = set(np.unique(x))
        if unique <= {0, 1}:
            a = int(np.sum((x == 1) & (y == 1)))
            b = int(np.sum((x == 1) & (y == 0)))
            c = int(np.sum((x == 0) & (y == 1)))
            dd = int(np.sum((x == 0) & (y == 0)))
            if b * c == 0:
                or_val, se_log = np.nan, np.nan
            else:
                or_val = (a * dd) / (b * c)
                se_log = np.sqrt(1 / max(a, 1) + 1 / max(b, 1) + 1 / max(c, 1) + 1 / max(dd, 1))
        else:
            # OLS approximation for log(OR) per unit increase
            from numpy.linalg import lstsq

            X = np.column_stack([np.ones(n), x])
            beta, _, _, _ = lstsq(X, y, rcond=None)
            b1 = beta[1]
            resid = y - X @ beta
            mse = np.sum(resid**2) / (n - 2)
            XtX_inv = np.linalg.pinv(X.T @ X)
            se_b = np.sqrt(mse * XtX_inv[1, 1])
            # Approximate log-OR from linear probability
            or_val = np.exp(b1 * 4) if abs(b1) < 5 else np.nan  # multiply by 4 for logit approx
            se_log = se_b * 4

        if np.isfinite(or_val) and np.isfinite(se_log) and se_log > 0:
            log_or = np.log(or_val) if or_val > 0 else np.nan
            ci_lo = np.exp(log_or - 1.96 * se_log) if np.isfinite(log_or) else np.nan
            ci_hi = np.exp(log_or + 1.96 * se_log) if np.isfinite(log_or) else np.nan
            z = log_or / se_log if np.isfinite(log_or) else 0
            pval = float(2 * (1 - stats.norm.cdf(abs(z))))
        else:
            ci_lo = ci_hi = pval = np.nan

        results.append(
            {
                "covariate": cov,
                "OR": round(float(or_val), 4) if np.isfinite(or_val) else np.nan,
                "ci_lower": round(float(ci_lo), 4) if np.isfinite(ci_lo) else np.nan,
                "ci_upper": round(float(ci_hi), 4) if np.isfinite(ci_hi) else np.nan,
                "pval": round(float(pval), 4) if np.isfinite(pval) else np.nan,
                "n": n,
            }
        )

    return pd.DataFrame(results)


def cheatsheet() -> str:
    return "otis_risk_table({}) -> Univariate risk factor table for OTIS correctional data."

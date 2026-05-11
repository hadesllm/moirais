# morie.fn — function file (hadesllm/morie)
"""Heterogeneous treatment effects test by region for OTIS data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy import stats


def otis_het_region(df: pd.DataFrame, cdf=None, *, outcome: str = "Y", treatment: str = "D", region_col: str = "region") -> dict:
    """Test for heterogeneous treatment effects across regions.

    Fits Y = a + b*D + c*region + d*(D*region) + e and tests
    whether the interaction coefficients are jointly significant
    (F-test on restricted vs unrestricted model).

    Parameters
    ----------
    df : DataFrame
        Data with outcome, treatment, and region columns.
    outcome, treatment, region_col : str
        Column names.

    Returns
    -------
    dict
        Keys: f_stat, pval, df1, df2, n, regions, interaction_coefs.
    """
    data = df[[outcome, treatment, region_col]].dropna().copy()
    dummies = pd.get_dummies(data[region_col], drop_first=True, dtype=float)
    regions = dummies.columns.tolist()

    y = data[outcome].values.astype(np.float64)
    d = data[treatment].values.astype(np.float64)
    R = dummies.values.astype(np.float64)
    n = len(y)

    # Restricted model: Y = a + b*D + c*R
    X_r = np.column_stack([np.ones(n), d, R])
    beta_r, _, _, _ = lstsq(X_r, y, rcond=None)
    ssr_r = float(np.sum((y - X_r @ beta_r) ** 2))

    # Unrestricted model: Y = a + b*D + c*R + d*(D*R)
    DR = d.reshape(-1, 1) * R
    X_u = np.column_stack([np.ones(n), d, R, DR])
    beta_u, _, _, _ = lstsq(X_u, y, rcond=None)
    ssr_u = float(np.sum((y - X_u @ beta_u) ** 2))

    q = DR.shape[1]  # number of interaction terms
    p_u = X_u.shape[1]
    df1 = q
    df2 = n - p_u

    if ssr_u == 0 or df2 <= 0:
        return {
            "f_stat": np.nan,
            "pval": np.nan,
            "df1": df1,
            "df2": df2,
            "n": n,
            "regions": regions,
            "interaction_coefs": {},
        }

    f_stat = float(((ssr_r - ssr_u) / df1) / (ssr_u / df2))
    pval = float(1 - stats.f.cdf(f_stat, df1, df2))

    # Extract interaction coefficients
    interaction_coefs = {}
    offset = 1 + 1 + R.shape[1]  # intercept + D + R columns
    for i, r in enumerate(regions):
        interaction_coefs[r] = float(beta_u[offset + i])

    return {
        "f_stat": f_stat,
        "pval": pval,
        "df1": df1,
        "df2": df2,
        "n": n,
        "regions": regions,
        "interaction_coefs": interaction_coefs,
    }


def cheatsheet() -> str:
    return "otis_het_region({}) -> Heterogeneous treatment effects test by region for OTIS data"

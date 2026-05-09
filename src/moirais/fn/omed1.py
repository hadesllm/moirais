# moirais.fn — function file (hadesllm/moirais)
"""Mediation analysis for OTIS correctional data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy import stats


def otis_mediation(df: pd.DataFrame, cdf=None, *, outcome: str = "Y", treatment: str = "D", mediator: str = "alert_mental_health", covariates: list[str] | None = None) -> dict:
    """Baron-Kenny mediation analysis.

    Estimates total, direct, and indirect (mediated) effects using
    the product-of-coefficients method with Sobel test.

    Parameters
    ----------
    df : DataFrame
        Data with outcome, treatment, mediator, and covariates.
    outcome, treatment, mediator : str
        Column names.
    covariates : list of str, optional
        Additional covariates.

    Returns
    -------
    dict
        Keys: total_effect, direct_effect, indirect_effect,
              sobel_z, sobel_pval, prop_mediated, n.
    """
    cols = [outcome, treatment, mediator]
    if covariates:
        cols += covariates
    data = df[cols].dropna().copy()
    n = len(data)

    y = data[outcome].values.astype(np.float64)
    d = data[treatment].values.astype(np.float64)
    m = data[mediator].values.astype(np.float64)

    if covariates and len(covariates) > 0:
        W = np.column_stack([data[c].values.astype(np.float64) for c in covariates])
    else:
        W = np.empty((n, 0))

    ones = np.ones((n, 1))

    # Path c: total effect (Y ~ D + W)
    Xc = np.column_stack([ones, d, W]) if W.shape[1] > 0 else np.column_stack([ones, d])
    beta_c, _, _, _ = lstsq(Xc, y, rcond=None)
    total = float(beta_c[1])

    # Path a: D -> M (M ~ D + W)
    Xa = Xc.copy()
    beta_a, _, _, _ = lstsq(Xa, m, rcond=None)
    a = float(beta_a[1])
    resid_a = m - Xa @ beta_a
    se_a = float(np.sqrt(np.sum(resid_a**2) / (n - Xa.shape[1]) / np.sum((d - d.mean()) ** 2)))

    # Path b + c': Y ~ D + M + W
    Xb = np.column_stack([ones, d, m, W]) if W.shape[1] > 0 else np.column_stack([ones, d, m])
    beta_b, _, _, _ = lstsq(Xb, y, rcond=None)
    direct = float(beta_b[1])  # c'
    b = float(beta_b[2])  # b
    resid_b = y - Xb @ beta_b
    mse_b = np.sum(resid_b**2) / (n - Xb.shape[1])
    XtX_inv = np.linalg.pinv(Xb.T @ Xb)
    se_b = float(np.sqrt(mse_b * XtX_inv[2, 2]))

    indirect = a * b

    # Sobel test
    sobel_se = np.sqrt(a**2 * se_b**2 + b**2 * se_a**2)
    sobel_z = float(indirect / sobel_se) if sobel_se > 0 else 0.0
    sobel_pval = float(2 * (1 - stats.norm.cdf(abs(sobel_z))))

    prop = float(indirect / total) if abs(total) > 1e-10 else np.nan

    return {
        "total_effect": total,
        "direct_effect": direct,
        "indirect_effect": indirect,
        "sobel_z": sobel_z,
        "sobel_pval": sobel_pval,
        "prop_mediated": prop,
        "n": n,
    }


def cheatsheet() -> str:
    return "otis_mediation({}) -> Mediation analysis for OTIS correctional data."

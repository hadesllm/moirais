# morie.fn — function file (hadesllm/morie)
"""IV estimation (2SLS) for OTIS correctional data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy import stats


def otis_iv_distance(df: pd.DataFrame, cdf=None, *, outcome: str = "Y", treatment: str = "D", instrument: str | None = None, covariates: list[str] | None = None) -> dict:
    """Estimate treatment effect via two-stage least squares (2SLS).

    If no instrument is specified, uses ``sentence_days`` as default
    (distance-to-facility or sentence length as exclusion restriction).

    Parameters
    ----------
    df : DataFrame
        Data with outcome, treatment, instrument, and covariates.
    outcome, treatment : str
        Column names.
    instrument : str, optional
        Instrumental variable column. Defaults to ``"sentence_days"``.
    covariates : list of str, optional
        Additional covariate columns.

    Returns
    -------
    dict
        Keys: iv_estimate, se, pval, ci_lower, ci_upper, n,
              first_stage_f, instrument.
    """
    if instrument is None:
        instrument = "sentence_days"

    cols = [outcome, treatment, instrument]
    if covariates:
        cols += covariates
    data = df[cols].dropna().copy()
    n = len(data)

    y = data[outcome].values.astype(np.float64)
    d = data[treatment].values.astype(np.float64)
    z_iv = data[instrument].values.astype(np.float64)

    if covariates and len(covariates) > 0:
        W = np.column_stack([np.ones(n)] + [data[c].values.astype(np.float64) for c in covariates])
    else:
        W = np.ones((n, 1))

    # Stage 1: D = gamma_0 + gamma_z * Z + gamma_w * W + u
    Z1 = np.column_stack([W, z_iv])
    gamma, _, _, _ = lstsq(Z1, d, rcond=None)
    d_hat = Z1 @ gamma

    # First-stage F-statistic for instrument
    resid_1 = d - d_hat
    ssr_r = float(np.sum((d - W @ lstsq(W, d, rcond=None)[0]) ** 2))
    ssr_u = float(np.sum(resid_1**2))
    k_diff = 1  # one instrument
    df_denom = n - Z1.shape[1]
    f_stat = float(((ssr_r - ssr_u) / k_diff) / (ssr_u / df_denom)) if ssr_u > 0 else 0.0

    # Stage 2: Y = beta_0 + beta_d * D_hat + beta_w * W + e
    X2 = np.column_stack([W, d_hat])
    beta, _, _, _ = lstsq(X2, y, rcond=None)
    iv_est = float(beta[-1])

    # Robust SE using original D residuals
    resid_2 = y - np.column_stack([W, d]) @ np.append(beta[:-1], iv_est)
    bread_inv = np.linalg.pinv(X2.T @ X2)
    meat = X2.T @ np.diag(resid_2**2) @ X2
    vcov = bread_inv @ meat @ bread_inv
    se = float(np.sqrt(vcov[-1, -1]))
    se = max(se, 1e-10)
    z = iv_est / se
    pval = float(2 * (1 - stats.norm.cdf(abs(z))))

    return {
        "iv_estimate": iv_est,
        "se": se,
        "pval": pval,
        "ci_lower": iv_est - 1.96 * se,
        "ci_upper": iv_est + 1.96 * se,
        "n": n,
        "first_stage_f": f_stat,
        "instrument": instrument,
    }


def cheatsheet() -> str:
    return "otis_iv_distance({}) -> IV estimation (2SLS) for OTIS correctional data."

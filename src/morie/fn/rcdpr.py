# morie.fn — function file (hadesllm/morie)
"""Logistic regression for recidivism risk factors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS


def recidivism_predictors(
    df: pd.DataFrame,
    *,
    outcome: str = DEFAULT_COLS["outcome"],
    covariates: list[str] | None = None,
) -> dict:
    """Logistic regression for recidivism risk factors.

    Binarizes outcome (>0 = recidivism) and fits logistic regression
    via iteratively reweighted least squares (IRLS).

    Parameters
    ----------
    df : DataFrame
        Dataset with outcome and covariate columns.
    outcome : str
        Column with recidivism outcome.
    covariates : list of str, optional
        Predictor column names. If None, uses all numeric columns
        except outcome.

    Returns
    -------
    dict
        coefficients, odds_ratios, p_values (dicts keyed by covariate),
        n_recid, n_total, pseudo_r2.
    """
    tmp = df.dropna(subset=[outcome])
    y = (tmp[outcome] > 0).astype(int).values

    if covariates is None:
        covariates = [c for c in tmp.select_dtypes(include="number").columns if c != outcome]
    if not covariates:
        return {
            "coefficients": {},
            "odds_ratios": {},
            "p_values": {},
            "n_recid": int(y.sum()),
            "n_total": len(y),
            "pseudo_r2": 0.0,
        }

    tmp = tmp.dropna(subset=covariates)
    y = (tmp[outcome] > 0).astype(int).values
    X = tmp[covariates].values.astype(float)
    X = np.column_stack([np.ones(len(X)), X])

    n_total = len(y)
    n_recid = int(y.sum())

    # IRLS for logistic regression
    beta = np.zeros(X.shape[1])
    for _ in range(25):
        eta = X @ beta
        eta = np.clip(eta, -20, 20)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = mu * (1.0 - mu)
        w = np.maximum(w, 1e-10)
        W = np.diag(w)
        z = eta + (y - mu) / w
        try:
            XtWX = X.T @ W @ X
            beta = np.linalg.solve(XtWX, X.T @ W @ z)
        except np.linalg.LinAlgError:
            break

    # Standard errors from Fisher information
    eta = X @ beta
    eta = np.clip(eta, -20, 20)
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = mu * (1.0 - mu)
    w = np.maximum(w, 1e-10)
    try:
        cov_mat = np.linalg.inv(X.T @ np.diag(w) @ X)
        se = np.sqrt(np.diag(cov_mat))
    except np.linalg.LinAlgError:
        se = np.full(len(beta), np.nan)

    from scipy import stats as _st

    z_vals = beta / np.where(se > 0, se, np.nan)
    p_vals = 2 * _st.norm.sf(np.abs(z_vals))

    # Null log-likelihood
    p_bar = y.mean()
    ll_null = n_total * (p_bar * np.log(p_bar + 1e-12) + (1 - p_bar) * np.log(1 - p_bar + 1e-12))
    ll_model = np.sum(y * np.log(mu + 1e-12) + (1 - y) * np.log(1 - mu + 1e-12))
    pseudo_r2 = 1.0 - ll_model / ll_null if ll_null != 0 else 0.0

    coef_names = ["intercept"] + list(covariates)
    coefficients = {n: float(b) for n, b in zip(coef_names, beta)}
    odds_ratios = {n: float(np.exp(b)) for n, b in zip(coef_names, beta)}
    p_values = {n: float(p) for n, p in zip(coef_names, p_vals)}

    return {
        "coefficients": coefficients,
        "odds_ratios": odds_ratios,
        "p_values": p_values,
        "n_recid": n_recid,
        "n_total": n_total,
        "pseudo_r2": float(pseudo_r2),
    }


rcdpr = recidivism_predictors


def cheatsheet() -> str:
    return "recidivism_predictors({}) -> Logistic regression for recidivism risk factors."

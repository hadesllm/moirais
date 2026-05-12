# morie.fn -- function file (hadesllm/morie)
"""Nuisance parameter estimation for causal inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._helpers import _validate_df


def nuisance_estimate(
    data: pd.DataFrame,
    *,
    y: str = "outcome",
    t: str = "treatment",
    covariates: list[str] | None = None,
    method: str = "ridge",
    lam: float = 1.0,
    seed: int = 42,
) -> dict:
    r"""Estimate nuisance functions for DML/AIPW.

    Estimates two nuisance functions:

    - Outcome model: :math:`\hat{m}(X) = E[Y|X]` via regression
    - Propensity score: :math:`\hat{e}(X) = P(T=1|X)` via logistic regression

    Supports ridge regression (no sklearn dependency).

    Parameters
    ----------
    data : pd.DataFrame
    y, t : str
        Outcome and treatment columns.
    covariates : list[str]
        Covariate columns for nuisance estimation.
    method : str
        'ridge' (default) or 'ols'.
    lam : float
        Ridge penalty parameter.
    seed : int
        Random seed (unused but kept for API consistency).

    Returns
    -------
    dict
        Keys: 'm_hat' (outcome predictions), 'e_hat' (PS predictions),
        'beta_m' (outcome coefficients), 'beta_e' (PS coefficients).

    References
    ----------
    Chernozhukov, V., et al. (2018). Double/debiased machine learning.
    *The Econometrics Journal*, 21(1), C1-C68.
    """
    if covariates is None or len(covariates) == 0:
        raise ValueError("covariates required")
    _validate_df(data, y, t, *covariates)
    df = data[[y, t] + covariates].dropna()
    n = len(df)

    Y = df[y].to_numpy(dtype=float)
    T_arr = df[t].to_numpy(dtype=float)
    X = df[covariates].to_numpy(dtype=float)
    X_design = np.column_stack([np.ones(n), X])
    p = X_design.shape[1]

    if method == "ridge":
        reg = lam * np.eye(p)
        reg[0, 0] = 0
        beta_m = np.linalg.solve(X_design.T @ X_design + reg, X_design.T @ Y)
    elif method == "ols":
        beta_m = np.linalg.lstsq(X_design, Y, rcond=None)[0]
    else:
        raise ValueError(f"Unknown method: {method}")

    m_hat = X_design @ beta_m

    beta_e = np.zeros(p)
    for _ in range(50):
        z = X_design @ beta_e
        phat = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        phat = np.clip(phat, 1e-8, 1 - 1e-8)
        W = phat * (1 - phat)
        grad = X_design.T @ (T_arr - phat)
        if method == "ridge":
            grad -= lam * beta_e
            H = X_design.T @ (X_design * W[:, None]) + lam * np.eye(p)
        else:
            H = X_design.T @ (X_design * W[:, None])
        try:
            beta_e += np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
    e_hat = 1.0 / (1.0 + np.exp(-np.clip(X_design @ beta_e, -500, 500)))
    e_hat = np.clip(e_hat, 0.01, 0.99)

    return {
        "m_hat": m_hat,
        "e_hat": e_hat,
        "beta_m": beta_m,
        "beta_e": beta_e,
        "n": n,
    }


nuis = nuisance_estimate


def cheatsheet() -> str:
    return "nuisance_estimate({}) -> Nuisance parameter estimation for causal inference."

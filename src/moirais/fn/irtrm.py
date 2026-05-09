# moirais.fn — function file (hadesllm/moirais)
"""Rasch model residuals (standardized)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def irt_rasch_residuals(
    data: pd.DataFrame | np.ndarray,
    item_params: dict,
    theta: np.ndarray,
) -> pd.DataFrame:
    """Standardized residuals for a Rasch (1PL) model.

    residual_ij = (x_ij - p_ij) / sqrt(p_ij * (1 - p_ij))

    Parameters
    ----------
    data : DataFrame or ndarray
        Binary item responses (n x k).
    item_params : dict
        {item_name: {'b': ...}} (Rasch difficulty).
    theta : ndarray
        Person ability estimates (n,).

    Returns
    -------
    DataFrame
        Standardized residuals (n x k), same shape as input.

    References
    ----------
    Wright, B. D. & Masters, G. N. (1982). Rating Scale Analysis.
    MESA Press.
    """
    X = np.asarray(data, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64).ravel()
    n, k = X.shape
    if len(theta) != n:
        raise ValueError(f"theta length {len(theta)} != n_persons {n}")

    params_list = list(item_params.values())
    if len(params_list) != k:
        raise ValueError(f"item_params has {len(params_list)} items, data has {k} columns")

    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"i{j}" for j in range(k)]

    residuals = np.full((n, k), np.nan)
    for j in range(k):
        b = params_list[j].get("b", 0.0)
        for i in range(n):
            if np.isnan(X[i, j]):
                continue
            z = theta[i] - b
            if z > 500:
                p = 1.0
            elif z < -500:
                p = 0.0
            else:
                p = 1.0 / (1.0 + np.exp(-z))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            residuals[i, j] = (X[i, j] - p) / np.sqrt(p * (1 - p))

    return pd.DataFrame(residuals, columns=names)


def cheatsheet() -> str:
    return "irt_rasch_residuals({}) -> Rasch model residuals (standardized)."

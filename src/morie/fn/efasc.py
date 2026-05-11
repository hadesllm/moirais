# morie.fn — function file (hadesllm/morie)
"""Compute factor scores from data and loadings."""

from __future__ import annotations

import numpy as np
import pandas as pd


def efa_scores(
    data: pd.DataFrame | np.ndarray,
    loadings: np.ndarray,
    *,
    method: str = "regression",
) -> np.ndarray:
    """Compute factor scores using the regression (Thompson) or Bartlett method.

    Parameters
    ----------
    data : DataFrame or ndarray
        Item responses (n x p).
    loadings : ndarray (p x n_factors)
        Factor loadings matrix (rotated or unrotated).
    method : str
        'regression' (Thomson, 1951) or 'bartlett' (Bartlett, 1937).
        Default 'regression'.

    Returns
    -------
    ndarray (n x n_factors)
        Factor scores.

    References
    ----------
    Thomson, G.H. (1951). The Factorial Analysis of Human Ability.
        University of London Press.
    Bartlett, M.S. (1937). The statistical conception of mental factors.
        British Journal of Psychology, 28, 97-104.
    """
    X = np.asarray(data, dtype=np.float64)
    L = np.asarray(loadings, dtype=np.float64)

    # Standardize X
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=1)
    sd[sd < 1e-10] = 1.0
    Z = (X - mu) / sd

    # Replace NaN with 0 after standardization
    Z = np.where(np.isfinite(Z), Z, 0.0)

    n, p = Z.shape

    R = np.corrcoef(Z, rowvar=False) if n > 1 else np.eye(p)

    if method == "regression":
        # Thompson scores: F = Z @ R^-1 @ L
        try:
            Rinv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            Rinv = np.linalg.pinv(R)
        return Z @ Rinv @ L

    elif method == "bartlett":
        # Bartlett scores: F = Z @ (L' @ Theta^-1 @ L)^-1 @ L' @ Theta^-1
        # Theta = diag(1 - communalities)
        comm = np.sum(L**2, axis=1)
        theta_diag = np.maximum(1 - comm, 0.01)
        Theta_inv = np.diag(1.0 / theta_diag)
        LtTi = L.T @ Theta_inv
        try:
            W = np.linalg.inv(LtTi @ L) @ LtTi
        except np.linalg.LinAlgError:
            W = np.linalg.pinv(LtTi @ L) @ LtTi
        return Z @ W.T

    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'regression' or 'bartlett'.")


def cheatsheet() -> str:
    return "efa_scores({}) -> Compute factor scores from data and loadings."

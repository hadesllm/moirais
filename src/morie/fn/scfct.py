# morie.fn — function file (hadesllm/morie)
"""Factor scores (regression method)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_factor(
    data: pd.DataFrame | np.ndarray,
    loadings: np.ndarray,
) -> np.ndarray:
    """Factor scores using the regression (Thomson) method.

    score = Z @ R_inv @ loadings, where Z is standardised data and
    R_inv is the inverse correlation matrix.

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.
    loadings : ndarray
        Factor loading matrix (p x n_factors).

    Returns
    -------
    ndarray
        Factor scores (n x n_factors).

    References
    ----------
    Thomson, G. H. (1951). The Factorial Analysis of Human Ability.
    University of London Press.
    """
    X = np.asarray(data, dtype=np.float64)
    L = np.asarray(loadings, dtype=np.float64)
    n, p = X.shape
    if L.shape[0] != p:
        raise ValueError(f"loadings rows {L.shape[0]} != number of items {p}")

    # Standardise
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd[sd < 1e-15] = 1.0
    Z = (X - mu) / sd

    # Correlation matrix
    R = np.corrcoef(X, rowvar=False)
    # Regularise for invertibility
    R += np.eye(p) * 1e-8
    R_inv = np.linalg.inv(R)

    return Z @ R_inv @ L


def cheatsheet() -> str:
    return "score_factor({}) -> Factor scores (regression method)."

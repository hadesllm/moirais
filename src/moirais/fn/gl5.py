# moirais.fn — function file (hadesllm/moirais)
"""Guttman's Lambda 5 reliability lower bound."""

from __future__ import annotations

import numpy as np
import pandas as pd


def gl5(
    data: pd.DataFrame | np.ndarray,
) -> float:
    """Guttman's Lambda 5 — lower bound using max item covariance sum.

    lambda_5 = lambda_1 + 2 * sqrt(max_j(sum_{i!=j} cov_ij^2)) / var_total

    For each item j, compute the sum of squared covariances with all
    other items, then take the maximum.  Lambda 5 is generally tighter
    than Lambda 1 but looser than Lambda 4.

    Parameters
    ----------
    data : DataFrame or ndarray
        Item matrix (respondents x items).

    Returns
    -------
    float
        Lambda 5 coefficient.

    References
    ----------
    Guttman, L. (1945). A basis for analyzing test-retest reliability.
    *Psychometrika*, 10(4), 255-282.
    """
    X = np.asarray(data, dtype=np.float64)
    mask = np.all(np.isfinite(X), axis=1)
    X = X[mask]
    n, k = X.shape

    if k < 2:
        return float("nan")

    C = np.cov(X, rowvar=False, ddof=1)
    total_var = C.sum()

    if total_var < 1e-15:
        return float("nan")

    item_var = np.diag(C)
    lam1 = 1.0 - item_var.sum() / total_var

    # For each item j, sum of squared covariances with all other items
    off_diag_sq = C**2
    np.fill_diagonal(off_diag_sq, 0.0)
    max_sq_sum = off_diag_sq.sum(axis=0).max()

    lam5 = lam1 + 2.0 * np.sqrt(max_sq_sum) / total_var

    return float(lam5)


short = gl5


def cheatsheet() -> str:
    return "gl5({}) -> Guttman's Lambda 5 reliability lower bound."

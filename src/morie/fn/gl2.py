# morie.fn — function file (hadesllm/morie)
"""Guttman's Lambda 2 reliability lower bound."""

from __future__ import annotations

import numpy as np
import pandas as pd


def gl2(
    data: pd.DataFrame | np.ndarray,
) -> float:
    """Guttman's Lambda 2 — improved lower bound using covariances.

    lambda_2 = lambda_1 + sqrt(k/(k-1) * sum_sq_cov)

    where sum_sq_cov = sum of squared off-diagonal covariances, and
    the square root term is added to lambda_1.  Lambda 2 >= Lambda 1
    always, and >= alpha when items are not tau-equivalent.

    Parameters
    ----------
    data : DataFrame or ndarray
        Item matrix (respondents x items).

    Returns
    -------
    float
        Lambda 2 coefficient.

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

    # Sum of squared off-diagonal covariances
    off_diag = C.copy()
    np.fill_diagonal(off_diag, 0.0)
    sum_sq_cov = np.sum(off_diag**2)

    adjustment = np.sqrt(k / (k - 1) * sum_sq_cov)
    lam2 = lam1 + adjustment / total_var

    return float(lam2)


short = gl2


def cheatsheet() -> str:
    return "gl2({}) -> Guttman's Lambda 2 reliability lower bound."

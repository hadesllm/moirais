# moirais.fn — function file (hadesllm/moirais)
"""Guttman's Lambda 6 reliability using squared multiple correlations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def gl6(
    data: pd.DataFrame | np.ndarray,
) -> float:
    """Guttman's Lambda 6 — reliability via squared multiple correlations.

    lambda_6 = 1 - sum(1 - R^2_j) / var_total

    where R^2_j is the squared multiple correlation of item j predicted
    from all other items.  This is often the tightest of the six
    Guttman bounds and is especially useful for multidimensional scales.

    Parameters
    ----------
    data : DataFrame or ndarray
        Item matrix (respondents x items).

    Returns
    -------
    float
        Lambda 6 coefficient.

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

    # Compute R^2_j for each item via the inverse of the correlation matrix.
    # R^2_j = 1 - 1/diag(R^{-1})_j
    R = np.corrcoef(X, rowvar=False)

    # Add small ridge to handle singular correlation matrices
    ridge = 1e-8 * np.eye(k)
    try:
        R_inv = np.linalg.inv(R + ridge)
    except np.linalg.LinAlgError:
        return float("nan")

    diag_inv = np.diag(R_inv)
    # Guard against zero/negative diagonal elements
    diag_inv = np.maximum(diag_inv, 1e-15)
    r_sq = 1.0 - 1.0 / diag_inv

    # Lambda 6: use item variances, not correlations
    item_var = np.diag(C)
    unique_var = item_var * (1.0 - r_sq)

    lam6 = 1.0 - unique_var.sum() / total_var

    return float(lam6)


short = gl6


def cheatsheet() -> str:
    return "gl6({}) -> Guttman's Lambda 6 reliability using squared multiple correl"

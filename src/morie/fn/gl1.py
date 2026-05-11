# morie.fn — function file (hadesllm/morie)
"""Guttman's Lambda 1 reliability lower bound."""

from __future__ import annotations

import numpy as np
import pandas as pd


def gl1(
    data: pd.DataFrame | np.ndarray,
) -> float:
    """Guttman's Lambda 1 — simplest lower bound on reliability.

    lambda_1 = 1 - sum(var_i) / var_total

    This is the proportion of total-score variance not attributable to
    individual item variances.  Always a lower bound on true reliability.

    Parameters
    ----------
    data : DataFrame or ndarray
        Item matrix (respondents x items).

    Returns
    -------
    float
        Lambda 1 coefficient.

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

    item_var = np.var(X, axis=0, ddof=1)
    total_var = np.var(X.sum(axis=1), ddof=1)

    if total_var < 1e-15:
        return float("nan")

    result = 1.0 - item_var.sum() / total_var
    return float(result)


short = gl1


def cheatsheet() -> str:
    return "gl1({}) -> Guttman's Lambda 1 reliability lower bound."

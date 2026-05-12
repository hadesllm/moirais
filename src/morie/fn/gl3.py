# morie.fn -- function file (hadesllm/morie)
"""Guttman's Lambda 3 (identical to Cronbach's alpha)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def gl3(
    data: pd.DataFrame | np.ndarray,
) -> float:
    """Guttman's Lambda 3 -- equivalent to Cronbach's coefficient alpha.

    lambda_3 = (k / (k-1)) * (1 - sum(var_i) / var_total)

    Included for completeness of the Guttman lambda family.  For the
    full alpha result with CI and standardised form, use ``crba()``.

    Parameters
    ----------
    data : DataFrame or ndarray
        Item matrix (respondents x items).

    Returns
    -------
    float
        Lambda 3 (alpha) coefficient.

    References
    ----------
    Guttman, L. (1945). A basis for analyzing test-retest reliability.
    *Psychometrika*, 10(4), 255-282.

    Cronbach, L. J. (1951). Coefficient alpha and the internal
    structure of tests. *Psychometrika*, 16(3), 297-334.
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

    result = (k / (k - 1)) * (1.0 - item_var.sum() / total_var)
    return float(result)


short = gl3


def cheatsheet() -> str:
    return "gl3({}) -> Guttman's Lambda 3 (identical to Cronbach's alpha)."

# morie.fn -- function file (hadesllm/morie)
"""Kuder-Richardson 20 reliability for dichotomous items."""

from __future__ import annotations

import numpy as np
import pandas as pd


def kr20(
    data: pd.DataFrame | np.ndarray,
) -> float:
    """Kuder-Richardson 20 reliability coefficient.

    KR-20 = (k / (k-1)) * (1 - sum(p_i * q_i) / var_total)

    Appropriate for tests with dichotomous (0/1) items.  This is the
    special case of Cronbach's alpha when all items are binary.

    Parameters
    ----------
    data : DataFrame or ndarray
        Binary item matrix (respondents x items), values 0 or 1.

    Returns
    -------
    float
        KR-20 reliability coefficient.

    Raises
    ------
    ValueError
        If any item contains values other than 0 and 1.

    References
    ----------
    Kuder, G. F., & Richardson, M. W. (1937). The theory of the
    estimation of test reliability. *Psychometrika*, 2(3), 151-160.
    """
    X = np.asarray(data, dtype=np.float64)
    # Drop rows with NaN
    mask = np.all(np.isfinite(X), axis=1)
    X = X[mask]
    n, k = X.shape

    if k < 2:
        return float("nan")

    # Validate binary
    unique = np.unique(X[np.isfinite(X)])
    if not np.all(np.isin(unique, [0.0, 1.0])):
        raise ValueError(f"KR-20 requires binary (0/1) items, got unique values: {unique}")

    p = X.mean(axis=0)  # item difficulty (proportion correct)
    q = 1.0 - p
    pq_sum = np.sum(p * q)

    total = X.sum(axis=1)
    var_total = np.var(total, ddof=1)

    if var_total < 1e-15:
        return float("nan")

    result = (k / (k - 1)) * (1.0 - pq_sum / var_total)
    return float(result)


short = kr20


def cheatsheet() -> str:
    return "kr20({}) -> Kuder-Richardson 20 reliability for dichotomous items."

# morie.fn — function file (hadesllm/morie)
"""Kuder-Richardson 21 reliability (equal difficulty assumption)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def kr21(
    data: pd.DataFrame | np.ndarray,
) -> float:
    """Kuder-Richardson 21 reliability coefficient.

    KR-21 = (k / (k-1)) * (1 - k * p_bar * q_bar / var_total)

    A simplified version of KR-20 that assumes all items have equal
    difficulty.  Always underestimates KR-20 when difficulties differ.

    Parameters
    ----------
    data : DataFrame or ndarray
        Binary item matrix (respondents x items), values 0 or 1.

    Returns
    -------
    float
        KR-21 reliability coefficient.

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
    mask = np.all(np.isfinite(X), axis=1)
    X = X[mask]
    n, k = X.shape

    if k < 2:
        return float("nan")

    unique = np.unique(X[np.isfinite(X)])
    if not np.all(np.isin(unique, [0.0, 1.0])):
        raise ValueError(f"KR-21 requires binary (0/1) items, got unique values: {unique}")

    total = X.sum(axis=1)
    mean_total = total.mean()
    var_total = np.var(total, ddof=1)

    if var_total < 1e-15:
        return float("nan")

    # p_bar = mean proportion correct across all items
    p_bar = mean_total / k
    q_bar = 1.0 - p_bar

    result = (k / (k - 1)) * (1.0 - (k * p_bar * q_bar) / var_total)
    return float(result)


short = kr21


def cheatsheet() -> str:
    return "kr21({}) -> Kuder-Richardson 21 reliability (equal difficulty assumption"

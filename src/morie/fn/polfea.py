"""Generate polynomial and interaction features."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def polynomial_features(
    data: np.ndarray,
    *,
    degree: int = 2,
    interaction_only: bool = False,
    include_bias: bool = False,
) -> DescriptiveResult:
    """Generate polynomial and interaction features.

    Parameters
    ----------
    data : ndarray of shape (n, p)
        Input features.
    degree : int
        Maximum polynomial degree.
    interaction_only : bool
        If True, only produce interaction terms (no x^2, x^3, ...).
    include_bias : bool
        If True, include a column of ones.

    Returns
    -------
    DescriptiveResult
        With ``value`` = expanded feature matrix (ndarray).
    """
    X = np.asarray(data, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if degree < 1:
        raise ValueError("degree must be >= 1")

    from itertools import combinations, combinations_with_replacement

    cols = [np.ones(n)] if include_bias else []

    for d in range(1, degree + 1):
        if interaction_only and d > 1:
            combos = combinations(range(p), d)
        else:
            combos = combinations_with_replacement(range(p), d)
        for combo in combos:
            col = np.ones(n)
            for idx in combo:
                col = col * X[:, idx]
            cols.append(col)

    result = np.column_stack(cols) if cols else np.empty((n, 0))

    return DescriptiveResult(
        name="polynomial_features",
        value=result,
        extra={
            "degree": degree,
            "n_features_in": p,
            "n_features_out": result.shape[1],
            "interaction_only": interaction_only,
        },
    )


polfea = polynomial_features


def cheatsheet() -> str:
    return 'polynomial_features({}) -> Feature engineering (polynomial expansion).'

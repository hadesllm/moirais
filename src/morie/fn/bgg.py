# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bootstrap aggregating (bagging). 'In time, the suffering of your people will persuade you.' -- Nute Gunray"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ._containers import DescriptiveResult


def _ols_predict(X_train: np.ndarray, y_train: np.ndarray, X_pred: np.ndarray) -> np.ndarray:
    """Simple OLS predict."""
    A = np.column_stack([np.ones(len(X_train)), X_train])
    beta, _, _, _ = np.linalg.lstsq(A, y_train, rcond=None)
    A_pred = np.column_stack([np.ones(len(X_pred)), X_pred])
    return A_pred @ beta


def bagging(
    X: np.ndarray,
    y: np.ndarray,
    n_bags: int = 50,
    predict_fn: Callable | None = None,
    seed: int = 42,
) -> DescriptiveResult:
    """Bootstrap aggregating (bagging).

    Parameters
    ----------
    X : ndarray
    y : ndarray
    n_bags : int, default 50
    predict_fn : callable or None
        ``predict_fn(X_train, y_train, X_pred) -> y_pred``.
        Defaults to OLS.
    seed : int, default 42

    Returns
    -------
    DescriptiveResult
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    rng = np.random.default_rng(seed)

    if predict_fn is None:
        predict_fn = _ols_predict

    all_preds = np.zeros((n_bags, n))
    for b in range(n_bags):
        idx = rng.choice(n, size=n, replace=True)
        all_preds[b] = predict_fn(X[idx], y[idx], X)

    bagged = np.mean(all_preds, axis=0)
    residuals = y - bagged
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return DescriptiveResult(
        name="Bagging",
        value=r_squared,
        extra={
            "predictions": bagged,
            "r_squared": r_squared,
            "n_bags": n_bags,
            "n": n,
        },
    )


bgg = bagging


def cheatsheet() -> str:
    return "_ols_predict({}) -> Bootstrap aggregating (bagging). 'In time, the suffering of "

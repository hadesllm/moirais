# moirais.fn — function file (hadesllm/moirais)
"""Cook's distance."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def cooks_distance(X: np.ndarray, y: np.ndarray) -> DescriptiveResult:
    """Cook's distance for OLS regression.

    Parameters
    ----------
    X : (n, p) predictor matrix
    y : (n,) response

    Returns
    -------
    DescriptiveResult
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    X_int = np.column_stack([np.ones(n), X])
    k = X_int.shape[1]

    beta = np.linalg.lstsq(X_int, y, rcond=None)[0]
    yhat = X_int @ beta
    resid = y - yhat
    mse = np.sum(resid**2) / (n - k)

    H = X_int @ np.linalg.inv(X_int.T @ X_int + 1e-10 * np.eye(k)) @ X_int.T
    h = np.diag(H)
    D = resid**2 * h / (k * mse * (1 - h) ** 2 + 1e-12)

    threshold = 4 / n
    influential = int(np.sum(threshold < D))

    return DescriptiveResult(
        name="cooks_distance",
        value=float(np.max(D)),
        extra={
            "distances": D.tolist(),
            "threshold_4_n": float(threshold),
            "n_influential": influential,
            "n": n,
            "k": k,
        },
    )


cooks = cooks_distance


def cheatsheet() -> str:
    return "cooks_distance({}) -> Cook's distance."

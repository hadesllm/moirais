# morie.fn — function file (hadesllm/morie)
"""DFFITS influence measure."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def dffits(X: np.ndarray, y: np.ndarray) -> DescriptiveResult:
    """DFFITS for each observation.

    Parameters
    ----------
    X : (n, p)
    y : (n,)

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

    XtX_inv = np.linalg.inv(X_int.T @ X_int + 1e-10 * np.eye(k))
    H = X_int @ XtX_inv @ X_int.T
    h = np.diag(H)
    beta = XtX_inv @ X_int.T @ y
    resid = y - X_int @ beta
    mse = np.sum(resid**2) / (n - k)

    s_i = np.sqrt(np.maximum((n - k) * mse - resid**2 / (1 - h + 1e-12), 1e-12) / (n - k - 1 + 1e-12))
    dffits_vals = resid * np.sqrt(h / (1 - h + 1e-12)) / (s_i + 1e-12)

    threshold = 2 * np.sqrt(k / n)
    n_influential = int(np.sum(np.abs(dffits_vals) > threshold))

    return DescriptiveResult(
        name="dffits",
        value=float(np.max(np.abs(dffits_vals))),
        extra={
            "dffits": dffits_vals.tolist(),
            "threshold": float(threshold),
            "n_influential": n_influential,
            "n": n,
            "k": k,
        },
    )


dffts = dffits


def cheatsheet() -> str:
    return "dffits({}) -> DFFITS influence measure."

# moirais.fn — function file (hadesllm/moirais)
"""DFBETAS influence measure."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def dfbetas(X: np.ndarray, y: np.ndarray) -> DescriptiveResult:
    """DFBETAS for each observation and coefficient.

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
    beta = XtX_inv @ X_int.T @ y
    resid = y - X_int @ beta
    H = X_int @ XtX_inv @ X_int.T
    h = np.diag(H)
    mse = np.sum(resid**2) / (n - k)

    dfb = np.zeros((n, k))
    for i in range(n):
        e_i = resid[i]
        s2_i = max(((n - k) * mse - e_i**2 / (1 - h[i] + 1e-12)) / (n - k - 1), 1e-12)
        c = XtX_inv @ X_int[i]
        dfb[i] = c * e_i / (1 - h[i] + 1e-12) / (np.sqrt(s2_i * np.diag(XtX_inv)) + 1e-12)

    threshold = 2 / np.sqrt(n)
    n_influential = int(np.sum(np.any(np.abs(dfb) > threshold, axis=1)))

    return DescriptiveResult(
        name="dfbetas",
        value=float(np.max(np.abs(dfb))),
        extra={
            "max_abs_dfbetas_per_coef": np.max(np.abs(dfb), axis=0).tolist(),
            "threshold": float(threshold),
            "n_influential": n_influential,
            "n": n,
            "k": k,
        },
    )


dfbts = dfbetas


def cheatsheet() -> str:
    return "dfbetas({}) -> DFBETAS influence measure."

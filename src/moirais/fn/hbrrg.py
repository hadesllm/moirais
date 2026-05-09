# moirais.fn — function file (hadesllm/moirais)
"""Huber robust regression."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def huber_regression(
    X,
    y,
    *,
    k: float = 1.345,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> ESRes:
    """Huber robust regression via iteratively reweighted least squares.

    Parameters
    ----------
    X : array-like, shape (n, p) or (n,)
        Design matrix (intercept NOT added automatically).
    y : array-like, shape (n,)
        Response.
    k : float
        Huber tuning constant (default 1.345).
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum IRLS iterations.

    Returns
    -------
    ESRes
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if n < p + 1:
        raise ValueError("Need n > p observations.")

    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    for _ in range(max_iter):
        resid = y - X @ beta
        s = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if s < 1e-12:
            break
        u = resid / s
        w = np.where(np.abs(u) <= k, 1.0, k / np.abs(u))
        W = np.diag(w)
        try:
            beta_new = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    resid = y - X @ beta
    s_final = 1.4826 * np.median(np.abs(resid))
    return ESRes(
        measure="huber_regression",
        estimate=float(np.sum(resid**2)),
        n=n,
        extra={
            "coefficients": beta.tolist(),
            "scale": float(s_final),
            "k": k,
            "p": p,
        },
    )


hbrrg = huber_regression


def cheatsheet() -> str:
    return "huber_regression(X, y) -> Huber robust regression."

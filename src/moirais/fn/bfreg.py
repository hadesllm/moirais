# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Biweight regression (Tukey bisquare)."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def biweight_regression(
    X,
    y,
    *,
    c: float = 4.685,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> ESRes:
    """Tukey biweight (bisquare) robust regression via IRLS.

    Parameters
    ----------
    X : array-like, shape (n, p) or (n,)
        Design matrix.
    y : array-like, shape (n,)
        Response.
    c : float
        Bisquare tuning constant (default 4.685).
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
        w = np.where(np.abs(u) <= c, (1.0 - (u / c) ** 2) ** 2, 0.0)
        W = np.diag(w)
        XtWX = X.T @ W @ X
        try:
            beta_new = np.linalg.solve(XtWX, X.T @ W @ y)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    resid = y - X @ beta
    s_final = 1.4826 * np.median(np.abs(resid))
    return ESRes(
        measure="biweight_regression",
        estimate=float(np.sum(resid**2)),
        n=n,
        extra={
            "coefficients": beta.tolist(),
            "scale": float(s_final),
            "c": c,
            "p": p,
        },
    )


bfreg = biweight_regression


def cheatsheet() -> str:
    return "biweight_regression(X, y) -> Tukey biweight robust regression."

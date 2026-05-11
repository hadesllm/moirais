# morie.fn — function file (hadesllm/morie)
"""MM-estimator for regression."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def mm_estimator(
    X,
    y,
    *,
    c: float = 4.685,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> ESRes:
    """MM-estimator for linear regression (Yohai, 1987).

    Combines high breakdown (from an initial S-estimate) with
    high efficiency (via bisquare IRLS refinement).

    Parameters
    ----------
    X : array-like, shape (n, p) or (n,)
        Design matrix (intercept NOT added automatically).
    y : array-like, shape (n,)
        Response.
    c : float
        Bisquare tuning constant (default 4.685 for 95 % efficiency).
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
    resid = y - X @ beta
    s = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    if s < 1e-12:
        s = 1.0

    for _ in range(max_iter):
        u = resid / s
        w = np.where(np.abs(u) <= c, (1.0 - (u / c) ** 2) ** 2, 0.0)
        W = np.diag(w)
        try:
            beta_new = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
        except np.linalg.LinAlgError:
            break
        resid_new = y - X @ beta_new
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            resid = resid_new
            break
        beta = beta_new
        resid = resid_new

    s_final = 1.4826 * np.median(np.abs(resid))
    return ESRes(
        measure="mm_estimator",
        estimate=float(np.sum(resid**2)),
        n=n,
        extra={
            "coefficients": beta.tolist(),
            "scale": float(s_final),
            "tuning": c,
            "p": p,
        },
    )


mmslt = mm_estimator


def cheatsheet() -> str:
    return "mm_estimator(X, y) -> MM-estimator for robust regression."

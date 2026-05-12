# morie.fn — function file (hadesllm/morie)
"""Penalised B-spline regression (P-splines; Eilers & Marx 1996).

Solves

    min_b  ||y - B b||^2 + lambda b' D' D b

where ``B`` is an equally-spaced B-spline basis of degree 3 and ``D`` is
the second-order difference matrix.  Closed-form solution

    b_hat = (B'B + lambda D'D)^{-1} B' y
"""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["penalized_spline"]


def _bspline_basis(x, knots, degree=3):
    """Cox-de Boor recursion (vectorised); returns (n, n_basis) array."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(knots, dtype=float)
    # number of basis functions = len(t) - degree - 1
    def N(i, k, t, x):
        if k == 0:
            return ((x >= t[i]) & (x < t[i + 1])).astype(float)
        denom1 = t[i + k] - t[i]
        denom2 = t[i + k + 1] - t[i + 1]
        a = ((x - t[i]) / denom1) * N(i, k - 1, t, x) if denom1 > 0 else 0
        b = ((t[i + k + 1] - x) / denom2) * N(i + 1, k - 1, t, x) if denom2 > 0 else 0
        return a + b
    n_basis = len(t) - degree - 1
    B = np.column_stack([N(i, degree, t, x) for i in range(n_basis)])
    # close at upper boundary
    B[x >= t[-degree - 1], -1] = 1.0
    return B


def penalized_spline(x, y, n_knots: int = 20, degree: int = 3,
                     lam: float = 1.0):
    """P-spline regression (Eilers-Marx).

    Parameters
    ----------
    x : (n,) array
    y : (n,) array
    n_knots : int
        Number of interior + boundary knots; default 20.
    degree : int
        B-spline degree (default 3, cubic).
    lam : float
        Penalty on 2nd differences of coefficients.

    Returns
    -------
    RichResult: coef, fitted, residuals, sse, r2, edf, lam, n, method.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    if n < degree + 2 or y.size != n:
        return RichResult(payload={"estimate": float("nan"), "n": int(n),
                                   "method": "P-spline (n too small)"})
    x_min, x_max = x.min(), x.max()
    if x_max == x_min:
        x_max = x_min + 1.0
    h = (x_max - x_min) / max(1, n_knots - 1)
    # extended knot vector for B-spline of given degree
    knots = np.concatenate([
        np.full(degree, x_min - h),
        np.linspace(x_min, x_max, n_knots),
        np.full(degree, x_max + h),
    ])
    B = _bspline_basis(x, knots, degree=degree)
    k = B.shape[1]
    D = np.diff(np.eye(k), n=2, axis=0)
    BtB = B.T @ B
    BtY = B.T @ y
    coef = np.linalg.solve(BtB + lam * (D.T @ D), BtY)
    fitted = B @ coef
    resid = y - fitted
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    # effective d.f. = trace of hat matrix
    H = B @ np.linalg.solve(BtB + lam * (D.T @ D), B.T)
    edf = float(np.trace(H))
    se = float(np.sqrt(sse / max(1, n - edf)) / np.sqrt(n))
    return RichResult(payload={
        "coef": coef, "fitted": fitted, "residuals": resid,
        "sse": sse, "r2": float(r2), "edf": edf, "lambda": float(lam),
        "estimate": float(fitted.mean()), "se": se, "n": int(n),
        "method": "P-spline (Eilers & Marx 1996)",
    })


# CANONICAL TEST
# >>> import numpy as np
# >>> rng = np.random.default_rng(0)
# >>> x = np.linspace(0, 1, 80)
# >>> y = np.sin(3 * x) + rng.normal(0, 0.05, 80)
# >>> res = penalized_spline(x, y, lam=0.1)
# >>> assert res["r2"] > 0.9


def cheatsheet():
    return "pspln(x, y, n_knots=20, lam=1.0): penalised B-spline (P-spline)."

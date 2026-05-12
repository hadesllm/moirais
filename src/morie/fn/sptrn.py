"""Polynomial trend surface analysis (OLS)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["spatial_trend_surface"]


def _design(coords, order):
    coords = np.atleast_2d(coords)
    n, d = coords.shape
    cols = [np.ones(n)]
    if d == 1:
        s = coords[:, 0]
        for k in range(1, order + 1):
            cols.append(s ** k)
    else:
        s1 = coords[:, 0]; s2 = coords[:, 1]
        if order >= 1:
            cols += [s1, s2]
        if order >= 2:
            cols += [s1 ** 2, s2 ** 2, s1 * s2]
        if order >= 3:
            cols += [s1 ** 3, s2 ** 3, s1 ** 2 * s2, s1 * s2 ** 2]
        if order >= 4:
            raise ValueError("trend_order > 3 not supported")
    return np.column_stack(cols)


def spatial_trend_surface(x, coords, order: int = 2):
    """
    OLS polynomial trend-surface fit:  mu(s) = sum_k beta_k f_k(s).

    Parameters
    ----------
    x : array-like, shape (n,)
    coords : array-like, shape (n, d), d in {1, 2}
    order : int, default 2 (allowed: 0, 1, 2, 3)

    Returns
    -------
    RichResult with payload: estimate (beta), se, r2, n, method.
    """
    y = np.asarray(x, dtype=float).ravel()
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    n = y.size
    if coords.shape[0] != n:
        raise ValueError(f"coords rows ({coords.shape[0]}) must match x ({n})")
    F = _design(coords, order)
    p = F.shape[1]
    if n < p:
        raise ValueError(f"need n >= {p} for trend_order={order}")
    XtX = F.T @ F
    beta = np.linalg.solve(XtX, F.T @ y)
    e = y - F @ beta
    sigma2 = float(e @ e) / max(n - p, 1)
    cov = sigma2 * np.linalg.inv(XtX)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = float(1.0 - (e @ e) / ss_tot) if ss_tot > 0 else 1.0
    return RichResult(payload={
        "estimate": beta.tolist(),
        "se": se.tolist(),
        "r2": r2,
        "order": int(order),
        "n": int(n),
        "method": f"Polynomial trend surface (order={order}, OLS)",
    })


def cheatsheet():
    return "sptrn: Trend-surface analysis (polynomial OLS)"


# CANONICAL TEST
# x = [1, 2, 3, 4, 5], coords = [[0],[1],[2],[3],[4]], order=1
# Expect beta = [1, 1] (intercept, slope), r2 = 1.0

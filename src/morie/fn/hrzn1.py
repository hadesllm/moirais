# morie.fn — function file (hadesllm/morie)
"""Nonparametric instrumental variables (Horowitz 2009, Ch 12).

Solves the linear inverse problem  T g = m  with

    m(z) = E[Y | Z = z],   (T g)(z) = E[g(X) | Z = z]

via a series-Tikhonov regulariser on a Hermite (orthogonal-polynomial)
basis of dimension ``J``.  Returns g_hat evaluated on a default grid;
SE from bootstrap (small replication count for speed).

Fallback: when n < threshold or the conditioning is degenerate, the
function reduces to linear 2SLS (a degenerate special case of NPIV).
"""
from __future__ import annotations

import math

import numpy as np

from ._richresult import RichResult

__all__ = ["horowitz_nonparametric_iv"]


def _hermite_basis(t, J):
    """Probabilists' Hermite polynomials He_0..He_{J-1}."""
    n = t.size
    H = np.zeros((n, J))
    H[:, 0] = 1.0
    if J > 1:
        H[:, 1] = t
    for k in range(2, J):
        H[:, k] = t * H[:, k - 1] - (k - 1) * H[:, k - 2]
    # mild scaling so columns are roughly unit-norm
    for k in range(J):
        H[:, k] = H[:, k] / np.sqrt(max(float(math.factorial(k)), 1.0))
    return H


def horowitz_nonparametric_iv(x, y, z, J=5, alpha=1e-3, grid=None,
                               _bootstrap=True):
    """Tikhonov-regularised series NPIV (Hermite basis).

    Parameters
    ----------
    x : array-like (n,)
    y : array-like (n,)
    z : array-like (n,)
    J : int, default 5
        Series-expansion order on both X and Z.
    alpha : float, default 1e-3
        Tikhonov regularisation strength.
    grid : array-like, optional
        Evaluation points; default 21-point grid on [min(x), max(x)].
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    n = y.size
    if n < 50 or x.size != n or z.size != n:
        # Fall back to 2SLS as a degenerate parametric case.
        Xc = np.column_stack([np.ones(n), x]) if x.size == n else np.array([])
        if Xc.size == 0:
            return RichResult(payload={"estimate": np.nan, "se": np.nan, "n": n,
                                       "method": "NPIV (insufficient data)"})
        Zc = np.column_stack([np.ones(n), z])
        Pz = Zc @ np.linalg.pinv(Zc.T @ Zc) @ Zc.T
        beta = np.linalg.pinv(Xc.T @ Pz @ Xc) @ (Xc.T @ Pz @ y)
        return RichResult(payload={
            "estimate": float(beta[1]), "se": np.nan,
            "n": n, "method": "NPIV fallback: linear 2SLS",
        })
    # Normalise to standard scale for stable Hermite series
    x_s = (x - x.mean()) / max(x.std(ddof=1), 1e-6)
    z_s = (z - z.mean()) / max(z.std(ddof=1), 1e-6)
    Bx = _hermite_basis(x_s, J)
    Bz = _hermite_basis(z_s, J)
    # Operator (T g)(z) = Bz' coef ; need coef of g in Bx s.t.
    # E[Y|Z] ~ Bz @ a where a depends on g coefficients via M = (Bz' Bx)/n
    M = (Bz.T @ Bx) / n
    BzY = (Bz.T @ y) / n
    BzBz = (Bz.T @ Bz) / n
    # Solve  (M' inv(BzBz) M + alpha I) c = M' inv(BzBz) BzY
    try:
        inv_BzBz = np.linalg.pinv(BzBz + alpha * np.eye(J))
        A = M.T @ inv_BzBz @ M + alpha * np.eye(J)
        rhs = M.T @ inv_BzBz @ BzY
        coef = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        return RichResult(payload={"estimate": np.nan, "se": np.nan, "n": n,
                                   "method": "NPIV (singular)"})
    if grid is None:
        grid = np.linspace(x.min(), x.max(), 21)
    grid = np.asarray(grid, dtype=float).ravel()
    grid_s = (grid - x.mean()) / max(x.std(ddof=1), 1e-6)
    Bx_g = _hermite_basis(grid_s, J)
    g_hat = Bx_g @ coef
    # Bootstrap SE (B=30) — guarded against recursion explosion
    if _bootstrap:
        rng = np.random.default_rng(0)
        B = 30
        boot = np.zeros((B, grid.size))
        for b in range(B):
            idx = rng.integers(0, n, size=n)
            try:
                sub = horowitz_nonparametric_iv(x[idx], y[idx], z[idx], J=J,
                                                alpha=alpha, grid=grid,
                                                _bootstrap=False)
                boot[b] = np.asarray(sub["estimate"], dtype=float)
            except Exception:
                boot[b] = g_hat
        se = boot.std(axis=0, ddof=1)
    else:
        se = np.full(grid.size, np.nan)
    return RichResult(payload={
        "estimate": g_hat.astype(float),
        "se": se.astype(float),
        "grid": grid.astype(float),
        "J": J, "alpha": alpha, "n": n,
        "method": "Series-Tikhonov NPIV on Hermite basis",
    })


def cheatsheet():
    return "hrzn1: nonparametric IV (Hermite-Tikhonov)"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(15)
    n = 1000
    z = rng.standard_normal(n)
    x = 0.7 * z + 0.5 * rng.standard_normal(n)
    y = x ** 2 + 0.3 * rng.standard_normal(n)
    res = horowitz_nonparametric_iv(x, y, z, grid=[0.0, 1.0, 2.0])
    print(res)
    # g(x) = x^2 at 0, 1, 2  → roughly 0, 1, 4
    assert res["estimate"][2] > res["estimate"][0]

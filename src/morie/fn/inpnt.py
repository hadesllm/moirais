# morie.fn — function file (hadesllm/morie)
"""Interior point method for linear programming."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def interior_point_lp(
    c: np.ndarray,
    A_eq: np.ndarray,
    b_eq: np.ndarray,
    *,
    tol: float = 1e-8,
    maxiter: int = 100,
) -> DescriptiveResult:
    """Primal-dual interior point method for LP.

    Solves: minimise c^T x subject to A_eq x = b_eq, x >= 0.

    Parameters
    ----------
    c : ndarray
        Objective coefficients.
    A_eq : ndarray
        Equality constraint matrix (m x n).
    b_eq : ndarray
        Equality constraint RHS.
    tol : float
        Duality gap tolerance.
    maxiter : int
        Maximum iterations.

    Returns
    -------
    DescriptiveResult
        ``value`` is the optimal objective; ``extra`` has x and iterations.
    """
    c = np.asarray(c, dtype=float)
    A = np.asarray(A_eq, dtype=float)
    b = np.asarray(b_eq, dtype=float)
    m, n = A.shape
    x = np.ones(n)
    lam = np.zeros(m)
    s = np.ones(n)
    for it in range(1, maxiter + 1):
        mu = (x @ s) / n
        if mu < tol:
            break
        sigma = 0.3
        X = np.diag(x)
        S = np.diag(s)
        rp = b - A @ x
        rd = c - A.T @ lam - s
        rc = sigma * mu * np.ones(n) - X @ S @ np.ones(n)
        # Use solve, not inv — numerically stable on ARM64 (ARM BLAS is
        # pickier about near-singular matrices than macOS Accelerate).
        XiS_reg = np.diag(s / (x + 1e-30)) + 1e-8 * np.eye(n)
        try:
            XiS_inv_AT = np.linalg.solve(XiS_reg, A.T)
        except np.linalg.LinAlgError:
            break
        M = A @ XiS_inv_AT + 1e-10 * np.eye(m)
        rhs = rp + A @ ((X @ rd + rc) / (s + 1e-30))
        try:
            dlam = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            break
        ds = rd + A.T @ dlam
        dx = (rc - X @ ds) / (s + 1e-30)
        alpha_p = 1.0
        alpha_d = 1.0
        for i in range(n):
            if dx[i] < 0:
                alpha_p = min(alpha_p, -0.99 * x[i] / dx[i])
            if ds[i] < 0:
                alpha_d = min(alpha_d, -0.99 * s[i] / ds[i])
        x += alpha_p * dx
        lam += alpha_d * dlam
        s += alpha_d * ds
    return DescriptiveResult(
        name="Interior Point LP",
        value=float(c @ x),
        extra={"x": x, "iterations": it, "duality_gap": float(x @ s)},
    )


inpnt = interior_point_lp

"""Sequential Quadratic Programming (SQP)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ._containers import DescriptiveResult


def sqp_optimize(
    f: Callable,
    grad_f: Callable,
    constraints: list[Callable],
    x0: np.ndarray,
    *,
    tol: float = 1e-6,
    maxiter: int = 100,
    h: float = 1e-7,
) -> DescriptiveResult:
    """Sequential Quadratic Programming for equality-constrained problems.

    Solves: min f(x) s.t. c_i(x) = 0.
    Uses BFGS approximation to the Hessian of the Lagrangian.

    Parameters
    ----------
    f : callable
        Objective function.
    grad_f : callable
        Gradient of the objective.
    constraints : list of callable
        Equality constraints c_i(x) = 0.
    x0 : ndarray
        Initial point.
    tol : float
        KKT tolerance.
    maxiter : int
        Maximum iterations.
    h : float
        Finite-difference step.

    Returns
    -------
    DescriptiveResult
        ``value`` is the final objective; ``extra`` has x and iterations.
    """
    x = np.asarray(x0, dtype=float).copy()
    n = len(x)
    nc = len(constraints)
    B = np.eye(n)
    lam = np.zeros(nc)
    for it in range(1, maxiter + 1):
        gf = np.asarray(grad_f(x), dtype=float)
        cv = np.array([constraints[k](x) for k in range(nc)])
        A_jac = np.zeros((nc, n))
        for k in range(nc):
            for j in range(n):
                xp = x.copy()
                xp[j] += h
                A_jac[k, j] = (constraints[k](xp) - cv[k]) / h
        if np.linalg.norm(cv) < tol and np.linalg.norm(gf + A_jac.T @ lam) < tol:
            break
        K = np.block([[B, A_jac.T], [A_jac, np.zeros((nc, nc))]])
        rhs = np.concatenate([-gf, -cv])
        try:
            sol = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
        dx = sol[:n]
        lam = sol[n:]
        x_new = x + dx
        s = dx
        gf_new = np.asarray(grad_f(x_new), dtype=float)
        y = gf_new - gf
        sy = s @ y
        if sy > 1e-15:
            Bs = B @ s
            B = B - np.outer(Bs, Bs) / (s @ Bs + 1e-30) + np.outer(y, y) / sy
        x = x_new
    return DescriptiveResult(
        name="SQP",
        value=float(f(x)),
        extra={
            "x": x,
            "iterations": it,
            "constraint_violation": float(np.linalg.norm(cv)),
        },
    )


sqprg = sqp_optimize

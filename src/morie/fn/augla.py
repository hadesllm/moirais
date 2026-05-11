# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Augmented Lagrangian method for constrained optimization."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ._containers import DescriptiveResult


def augmented_lagrangian(
    f: Callable,
    grad_f: Callable,
    constraints: list[Callable],
    x0: np.ndarray,
    *,
    rho: float = 1.0,
    tol: float = 1e-6,
    maxiter: int = 50,
    inner_maxiter: int = 200,
    inner_lr: float = 0.01,
) -> DescriptiveResult:
    """Augmented Lagrangian method for equality-constrained optimization.

    Solves: min f(x) s.t. h_i(x) = 0 for each constraint h_i.

    Parameters
    ----------
    f : callable
        Objective function.
    grad_f : callable
        Gradient of the objective.
    constraints : list of callable
        Each h_i(x) -> scalar; the constraint is h_i(x) = 0.
    x0 : ndarray
        Initial point.
    rho : float
        Penalty parameter (increased each outer iteration).
    tol : float
        Constraint violation tolerance.
    maxiter : int
        Maximum outer iterations.
    inner_maxiter : int
        Inner gradient steps per outer iteration.
    inner_lr : float
        Inner gradient descent learning rate.

    Returns
    -------
    DescriptiveResult
        ``value`` is the final objective; ``extra`` has x, multipliers,
        and constraint violation.
    """
    x = np.asarray(x0, dtype=float).copy()
    nc = len(constraints)
    lam = np.zeros(nc)
    for outer in range(1, maxiter + 1):
        for _ in range(inner_maxiter):
            g = np.asarray(grad_f(x), dtype=float)
            h = 1e-7
            for k in range(nc):
                ck = constraints[k](x)
                cg = np.zeros_like(x)
                for j in range(len(x)):
                    xp = x.copy()
                    xp[j] += h
                    cg[j] = (constraints[k](xp) - ck) / h
                g += (lam[k] + rho * ck) * cg
            x -= inner_lr * g
        cv = np.array([constraints[k](x) for k in range(nc)])
        for k in range(nc):
            lam[k] += rho * cv[k]
        if np.linalg.norm(cv) < tol:
            break
        rho *= 2.0
    return DescriptiveResult(
        name="Augmented Lagrangian",
        value=float(f(x)),
        extra={
            "x": x,
            "multipliers": lam,
            "constraint_violation": float(np.linalg.norm(cv)),
            "outer_iterations": outer,
        },
    )


augla = augmented_lagrangian

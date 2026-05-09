# moirais.fn — function file (hadesllm/moirais)
"""Conjugate gradient descent optimizer."""

import numpy as np

from ._containers import DescriptiveResult


def conjugate_gradient(f, grad_f, x0, max_iter=200, tol=1e-8):
    """
    Nonlinear conjugate gradient method (Fletcher-Reeves).

    :param f: Objective function f(x) -> scalar.
    :param grad_f: Gradient function grad_f(x) -> array.
    :param x0: (d,) initial point.
    :param max_iter: Maximum iterations.
    :param tol: Convergence tolerance on gradient norm.
    :return: DescriptiveResult with optimal x, f(x), gradient norm history.

    References
    ----------
    Nocedal J, Wright SJ (2006). Numerical Optimization. 2nd ed.
    Springer. Chapter 5.
    """
    x = np.asarray(x0, dtype=np.float64).ravel().copy()
    g = grad_f(x)
    d = -g.copy()
    grad_norms = [float(np.linalg.norm(g))]

    for it in range(max_iter):
        if np.linalg.norm(g) < tol:
            break

        alpha = 1.0
        fx = f(x)
        for _ in range(30):
            x_new = x + alpha * d
            if f(x_new) < fx - 1e-4 * alpha * np.dot(g, d):
                break
            alpha *= 0.5

        x = x + alpha * d
        g_new = grad_f(x)
        beta = np.dot(g_new, g_new) / max(np.dot(g, g), 1e-300)
        d = -g_new + beta * d
        g = g_new
        grad_norms.append(float(np.linalg.norm(g)))

    return DescriptiveResult(
        name="conjugate_gradient",
        value=float(f(x)),
        extra={
            "x_opt": x.tolist(),
            "f_opt": float(f(x)),
            "n_iter": len(grad_norms) - 1,
            "final_grad_norm": grad_norms[-1],
            "converged": grad_norms[-1] < tol,
        },
    )


def cheatsheet() -> str:
    return "conjugate_gradient({}) -> Conjugate gradient descent optimizer."

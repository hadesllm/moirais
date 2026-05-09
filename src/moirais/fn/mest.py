# moirais.fn — function file (hadesllm/moirais)
"""M-estimator: argmin sum rho(X_i, theta)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize, stats

__all__ = ["mest"]


def mest(
    x: np.ndarray,
    rho: callable,
    *,
    theta0: float | np.ndarray | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""
    Compute an M-estimator: :math:`\hat\theta = \arg\min_\theta \sum_{i=1}^n \rho(X_i, \theta)`.

    Variance via sandwich estimator using numerical Hessian and gradient.

    :param x: Observation array, shape (n,) or (n, p).
    :param rho: Loss function rho(x_i, theta) -> scalar.
    :param theta0: Initial value. Default 0 or zeros.
    :param alpha: Significance level. Default 0.05.
    :return: Dict with ``theta``, ``se``, ``ci_lower``, ``ci_upper``,
        ``objective``, ``n``, ``converged``.
    :raises ValueError: If x is empty.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 8-9. Springer.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("x must be non-empty.")

    n = x.shape[0]
    scalar = theta0 is None or np.isscalar(theta0)

    if theta0 is None:
        theta0 = 0.0

    def objective(theta):
        return np.sum([rho(x[i], theta) for i in range(n)])

    if scalar:
        result = optimize.minimize_scalar(objective, bracket=(theta0 - 10, theta0 + 10))
        theta_hat = float(result.x)
        converged = result.success
        obj_val = float(result.fun)
    else:
        theta0 = np.asarray(theta0, dtype=float)
        result = optimize.minimize(objective, theta0, method="BFGS")
        theta_hat = result.x
        converged = result.success
        obj_val = float(result.fun)

    eps = 1e-5
    if scalar:
        grads = np.array([(rho(x[i], theta_hat + eps) - rho(x[i], theta_hat - eps)) / (2 * eps) for i in range(n)])
        B = float(np.mean(grads ** 2))
        hess_vals = np.array([(rho(x[i], theta_hat + eps) - 2 * rho(x[i], theta_hat) + rho(x[i], theta_hat - eps)) / eps ** 2 for i in range(n)])
        A = float(np.mean(hess_vals))
        if abs(A) < 1e-12:
            A = 1.0
        var = B / (A ** 2 * n)
        se = float(np.sqrt(max(var, 0)))
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        return {
            "theta": theta_hat,
            "se": se,
            "ci_lower": theta_hat - z * se,
            "ci_upper": theta_hat + z * se,
            "objective": obj_val,
            "n": n,
            "converged": converged,
        }
    else:
        theta_hat = np.asarray(theta_hat)
        d = theta_hat.size
        grads = np.zeros((n, d))
        for i in range(n):
            for j in range(d):
                e = np.zeros(d)
                e[j] = eps
                grads[i, j] = (rho(x[i], theta_hat + e) - rho(x[i], theta_hat - e)) / (2 * eps)
        B = grads.T @ grads / n
        A = np.zeros((d, d))
        for j in range(d):
            for k in range(d):
                ej = np.zeros(d)
                ek = np.zeros(d)
                ej[j] = eps
                ek[k] = eps
                A[j, k] = np.mean([(rho(x[i], theta_hat + ej + ek) - rho(x[i], theta_hat + ej - ek) - rho(x[i], theta_hat - ej + ek) + rho(x[i], theta_hat - ej - ek)) / (4 * eps ** 2) for i in range(n)])
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(A)
        V = A_inv @ B @ A_inv.T / n
        se = np.sqrt(np.maximum(np.diag(V), 0))
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        return {
            "theta": theta_hat,
            "se": se,
            "ci_lower": theta_hat - z * se,
            "ci_upper": theta_hat + z * se,
            "objective": obj_val,
            "n": n,
            "converged": converged,
        }


def cheatsheet() -> str:
    return "mest({x, rho}) -> M-estimator via loss minimization."

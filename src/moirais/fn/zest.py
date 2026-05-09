"""Z-estimator: solve sum psi(X_i, theta) = 0."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize, stats

__all__ = ["zest"]


def zest(
    x: np.ndarray,
    psi: callable,
    *,
    theta0: np.ndarray | float | None = None,
    alpha: float = 0.05,
    psi_deriv: callable | None = None,
) -> dict[str, Any]:
    r"""
    Compute a Z-estimator by solving :math:`\sum_{i=1}^n \psi(X_i, \theta) = 0`.

    Uses Newton-Raphson (if derivative provided) or Brent/fsolve otherwise.
    Variance estimated via sandwich: :math:`A^{-1} B (A^{-1})^T / n`
    where :math:`A = E[\dot\psi]`, :math:`B = E[\psi \psi^T]`.

    :param x: Observation array, shape (n,) or (n, p).
    :param psi: Estimating function psi(x_i, theta) -> scalar or array.
    :param theta0: Initial value. Default 0 (scalar) or zeros.
    :param alpha: Significance level. Default 0.05.
    :param psi_deriv: Derivative of psi w.r.t. theta. If None, uses numerical diff.
    :return: Dict with ``theta``, ``se``, ``ci_lower``, ``ci_upper``, ``n``,
        ``converged``.
    :raises ValueError: If x is empty.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 8. Springer.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("x must be non-empty.")
    if x.ndim == 1:
        x_obs = x
    else:
        x_obs = x

    n = x_obs.shape[0] if x_obs.ndim > 0 else 1
    scalar = theta0 is None or np.isscalar(theta0)

    if theta0 is None:
        theta0 = 0.0

    def sum_psi(theta):
        val = np.sum([psi(x_obs[i], theta) for i in range(n)], axis=0)
        return val

    try:
        if scalar:
            result = optimize.brentq(lambda t: float(sum_psi(t)), -100, 100)
            theta_hat = float(result)
            converged = True
        else:
            theta0 = np.asarray(theta0, dtype=float)
            sol = optimize.fsolve(sum_psi, theta0, full_output=True)
            theta_hat = sol[0]
            converged = sol[2] == 1
    except (ValueError, RuntimeError):
        if scalar:
            sol = optimize.minimize_scalar(lambda t: float(sum_psi(t)) ** 2)
            theta_hat = float(sol.x)
            converged = sol.success
        else:
            theta0 = np.asarray(theta0, dtype=float)
            sol = optimize.minimize(lambda t: float(np.sum(np.array(sum_psi(t)) ** 2)), theta0)
            theta_hat = sol.x
            converged = sol.success

    psi_vals = np.array([psi(x_obs[i], theta_hat) for i in range(n)])
    if psi_vals.ndim == 1:
        B = float(np.mean(psi_vals ** 2))
    else:
        B = psi_vals.T @ psi_vals / n

    eps = 1e-5
    if scalar:
        if psi_deriv is not None:
            A = float(np.mean([psi_deriv(x_obs[i], theta_hat) for i in range(n)]))
        else:
            sp = np.mean([psi(x_obs[i], theta_hat + eps) for i in range(n)])
            sm = np.mean([psi(x_obs[i], theta_hat - eps) for i in range(n)])
            A = (sp - sm) / (2 * eps)
        if abs(A) < 1e-12:
            A = 1.0
        var = B / (A ** 2 * n)
        se = float(np.sqrt(var))
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        return {
            "theta": theta_hat,
            "se": se,
            "ci_lower": theta_hat - z * se,
            "ci_upper": theta_hat + z * se,
            "n": n,
            "converged": converged,
        }
    else:
        theta_hat = np.asarray(theta_hat)
        d = theta_hat.size
        if psi_deriv is not None:
            A = np.mean([psi_deriv(x_obs[i], theta_hat) for i in range(n)], axis=0)
        else:
            A = np.zeros((d, d))
            for j in range(d):
                e = np.zeros(d)
                e[j] = eps
                sp = np.mean([psi(x_obs[i], theta_hat + e) for i in range(n)], axis=0)
                sm = np.mean([psi(x_obs[i], theta_hat - e) for i in range(n)], axis=0)
                A[:, j] = (sp - sm) / (2 * eps)
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(A)
        V = A_inv @ B @ A_inv.T / n
        se = np.sqrt(np.diag(V))
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        return {
            "theta": theta_hat,
            "se": se,
            "ci_lower": theta_hat - z * se,
            "ci_upper": theta_hat + z * se,
            "n": n,
            "converged": converged,
        }


def cheatsheet() -> str:
    return "zest({x, psi}) -> Z-estimator via estimating equations."

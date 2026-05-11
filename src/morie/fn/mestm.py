# morie.fn — function file (hadesllm/morie)
"""M-estimator with influence function and robust variance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MEstimatorResult:
    """Result of M-estimation.

    Attributes
    ----------
    theta : np.ndarray
        Estimated parameter vector.
    se : np.ndarray
        Standard errors (sandwich estimator).
    cov : np.ndarray
        Sandwich covariance matrix.
    influence_fn : np.ndarray
        Estimated influence function values for each observation, shape (n, p).
    n_iter : int
        Number of optimization iterations.
    converged : bool
        Whether the solver converged.
    n : int
        Sample size.
    """

    theta: np.ndarray
    se: np.ndarray
    cov: np.ndarray
    influence_fn: np.ndarray
    n_iter: int
    converged: bool
    n: int


def mestm(
    rho: object,
    psi: object,
    X: np.ndarray,
    *,
    theta0: np.ndarray | None = None,
    dpsi: object | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    step_size: float = 1.0,
) -> MEstimatorResult:
    r"""
    Compute an M-estimator with influence function and sandwich variance.

    An M-estimator minimizes the empirical criterion:

    .. math::

        \hat{\theta}_n = \arg\min_\theta \frac{1}{n}
        \sum_{i=1}^n \rho(X_i, \theta)

    Setting :math:`\psi = \nabla_\theta \rho`, the estimator solves
    :math:`P_n \psi(\cdot, \theta) = 0`. The influence function is:

    .. math::

        \mathrm{IF}(x; \hat{\theta}, F)
        = -\bigl[E[\dot{\psi}(X, \theta_0)]\bigr]^{-1} \psi(x, \theta_0)

    The sandwich covariance is:

    .. math::

        V = A^{-1} B (A^{-1})^\top, \quad
        A = E[\dot{\psi}], \quad B = E[\psi\psi^\top]

    When ``dpsi`` is not provided, a finite-difference Jacobian is used.

    :param rho: Objective function ``rho(x_i, theta) -> float`` (for reference;
        optimization uses psi).
    :param psi: Score function ``psi(x_i, theta) -> np.ndarray`` of shape (p,).
    :param X: Data matrix (n, d) or 1-D array.
    :param theta0: Starting value. Default zeros.
    :param dpsi: Jacobian of psi. ``dpsi(x_i, theta) -> np.ndarray`` (p, p).
        If None, computed via finite differences.
    :param max_iter: Maximum iterations. Default 100.
    :param tol: Convergence tolerance. Default 1e-8.
    :param step_size: Step size for Newton updates. Default 1.0.
    :return: MEstimatorResult with theta, SEs, covariance, and influence function.
    :raises ValueError: If X is empty.

    References
    ----------
    Kosorok, M.R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*, Ch. 13 (M-estimation). Springer.
    DOI:10.1007/978-0-387-74978-5

    Huber, P.J. (1964). Robust estimation of a location parameter.
    *Annals of Mathematical Statistics*, 35(1), 73--101.

    Hampel, F.R. (1974). The influence curve and its role in robust estimation.
    *JASA*, 69(346), 383--393.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    if n == 0:
        raise ValueError("X must be non-empty.")

    test_val = psi(X[0], np.zeros(1) if theta0 is None else theta0)
    p = np.asarray(test_val).size

    theta = np.zeros(p) if theta0 is None else np.asarray(theta0, dtype=float).copy()

    h = 1e-7

    def _fd_jacobian(xi, th):
        """Finite-difference Jacobian of psi."""
        jac = np.zeros((p, p))
        f0 = np.asarray(psi(xi, th)).ravel()
        for j in range(p):
            th_p = th.copy()
            th_p[j] += h
            f1 = np.asarray(psi(xi, th_p)).ravel()
            jac[:, j] = (f1 - f0) / h
        return jac

    jac_fn = dpsi if dpsi is not None else _fd_jacobian

    converged = False
    n_iter = 0

    for it in range(max_iter):
        psi_sum = np.zeros(p)
        for i in range(n):
            psi_sum += np.asarray(psi(X[i], theta)).ravel()
        psi_n = psi_sum / n

        if np.linalg.norm(psi_n) < tol:
            converged = True
            n_iter = it + 1
            break

        A_n = np.zeros((p, p))
        for i in range(n):
            A_n += np.asarray(jac_fn(X[i], theta)).reshape(p, p)
        A_n /= n

        try:
            delta = np.linalg.solve(A_n, psi_n)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(A_n, psi_n, rcond=None)[0]

        theta = theta - step_size * delta
        n_iter = it + 1

    A_hat = np.zeros((p, p))
    B_hat = np.zeros((p, p))
    psi_vals = np.zeros((n, p))
    for i in range(n):
        psi_i = np.asarray(psi(X[i], theta)).ravel()
        psi_vals[i] = psi_i
        A_hat += np.asarray(jac_fn(X[i], theta)).reshape(p, p)
        B_hat += np.outer(psi_i, psi_i)
    A_hat /= n
    B_hat /= n

    try:
        A_inv = np.linalg.inv(A_hat)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A_hat)

    cov = A_inv @ B_hat @ A_inv.T / n
    se = np.sqrt(np.diag(cov))

    influence = -psi_vals @ A_inv.T

    return MEstimatorResult(
        theta=theta,
        se=se,
        cov=cov,
        influence_fn=influence,
        n_iter=n_iter,
        converged=converged,
        n=n,
    )


def cheatsheet() -> str:
    return "mestm({rho, psi, X}) -> M-estimator with influence function."

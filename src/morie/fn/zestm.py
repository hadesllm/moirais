"""Z-estimator (estimating equations) with sandwich variance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ZEstimatorResult:
    """Result of Z-estimation.

    Attributes
    ----------
    theta : np.ndarray
        Estimated parameter vector.
    se : np.ndarray
        Standard errors (sandwich estimator).
    cov : np.ndarray
        Sandwich covariance matrix.
    n_iter : int
        Number of Newton-Raphson iterations used.
    converged : bool
        Whether the solver converged.
    n : int
        Sample size.
    """

    theta: np.ndarray
    se: np.ndarray
    cov: np.ndarray
    n_iter: int
    converged: bool
    n: int


def zestm(
    psi: object,
    dpsi: object,
    X: np.ndarray,
    *,
    theta0: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> ZEstimatorResult:
    r"""
    Compute a Z-estimator by solving estimating equations with sandwich variance.

    A Z-estimator :math:`\hat{\theta}_n` solves the estimating equation:

    .. math::

        \Psi_n(\theta) = \frac{1}{n} \sum_{i=1}^n \psi(X_i, \theta) = 0

    The asymptotic distribution is:

    .. math::

        \sqrt{n}(\hat{\theta}_n - \theta_0)
        \xrightarrow{d} N\bigl(0,\, A^{-1} B (A^{-1})^\top\bigr)

    where :math:`A = E[\dot{\psi}(X, \theta_0)]` (negative Jacobian) and
    :math:`B = E[\psi(X, \theta_0)\psi(X, \theta_0)^\top]` (meat).

    This is the sandwich variance estimator, robust to model misspecification.

    :param psi: Estimating function. Callable ``psi(x_i, theta) -> np.ndarray``
        of shape (p,) for one observation x_i.
    :param dpsi: Jacobian of psi. Callable ``dpsi(x_i, theta) -> np.ndarray``
        of shape (p, p) for one observation.
    :param X: Data matrix (n, d) or 1-D array of observations.
    :param theta0: Starting value for theta. If None, uses zeros.
    :param max_iter: Maximum Newton-Raphson iterations. Default 100.
    :param tol: Convergence tolerance on the norm of Psi_n. Default 1e-8.
    :return: ZEstimatorResult with estimated theta, SEs, and covariance.
    :raises ValueError: If X is empty.

    References
    ----------
    Kosorok, M.R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*, Ch. 13 (Z-estimators). Springer.
    DOI:10.1007/978-0-387-74978-5

    van der Vaart, A.W. (1998). *Asymptotic Statistics*, Ch. 5
    (M- and Z-estimators). Cambridge University Press.
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

    converged = False
    n_iter = 0

    for it in range(max_iter):
        psi_sum = np.zeros(p)
        for i in range(n):
            psi_sum += np.asarray(psi(X[i], theta))
        psi_n = psi_sum / n

        if np.linalg.norm(psi_n) < tol:
            converged = True
            n_iter = it + 1
            break

        dpsi_sum = np.zeros((p, p))
        for i in range(n):
            dpsi_sum += np.asarray(dpsi(X[i], theta)).reshape(p, p)
        A_n = dpsi_sum / n

        try:
            delta = np.linalg.solve(A_n, psi_n)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(A_n, psi_n, rcond=None)[0]

        theta = theta - delta
        n_iter = it + 1

    A_hat = np.zeros((p, p))
    B_hat = np.zeros((p, p))
    for i in range(n):
        psi_i = np.asarray(psi(X[i], theta)).ravel()
        dpsi_i = np.asarray(dpsi(X[i], theta)).reshape(p, p)
        A_hat += dpsi_i
        B_hat += np.outer(psi_i, psi_i)
    A_hat /= n
    B_hat /= n

    try:
        A_inv = np.linalg.inv(A_hat)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A_hat)

    cov = A_inv @ B_hat @ A_inv.T / n
    se = np.sqrt(np.diag(cov))

    return ZEstimatorResult(
        theta=theta,
        se=se,
        cov=cov,
        n_iter=n_iter,
        converged=converged,
        n=n,
    )


def cheatsheet() -> str:
    return "zestm({psi, dpsi, X}) -> Z-estimator with sandwich variance."

# morie.fn -- function file (hadesllm/morie)
"""2-Parameter Logistic IRT model via marginal MLE (EM)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from morie.fn._containers import IRTResult


def _icc_2pl(theta: np.ndarray, a: float, b: float) -> np.ndarray:
    """Item characteristic curve for 2PL."""
    logit = np.clip(a * (theta - b), -700, 700)
    return 1.0 / (1.0 + np.exp(-logit))


def irt2p(
    data: pd.DataFrame | np.ndarray,
    *,
    n_quad: int = 41,
    max_iter: int = 200,
    tol: float = 1e-5,
    theta_grid: np.ndarray | None = None,
) -> IRTResult:
    """Fit a 2-Parameter Logistic IRT model via marginal MLE (EM).

    Estimates item discrimination (a) and difficulty (b) parameters using
    an EM algorithm with Gauss-Hermite quadrature for the E-step.

    Parameters
    ----------
    data : DataFrame or ndarray
        Binary item response matrix (n x k), values 0/1.
    n_quad : int
        Number of quadrature points (default 41).
    max_iter : int
        Maximum EM iterations (default 200).
    tol : float
        Convergence tolerance on marginal log-likelihood (default 1e-5).
    theta_grid : ndarray, optional
        Grid for test information.  Default linspace(-4, 4, 81).

    Returns
    -------
    IRTResult
        model="2PL", item_params={item: {"a": float, "b": float}}.

    References
    ----------
    Bock, R. D. & Aitkin, M. (1981). Marginal maximum likelihood estimation
    of item parameters. Psychometrika, 46(4), 443-459.
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape

    if k < 2:
        raise ValueError("Need at least 2 items for IRT model.")

    X = np.where(np.isnan(X), 0.0, X)

    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"item_{j}" for j in range(k)]

    # Gauss-Hermite quadrature points and weights
    quad_pts, quad_wts = np.polynomial.hermite.hermgauss(n_quad)
    quad_pts = quad_pts * np.sqrt(2)  # scale for N(0,1)
    quad_wts = quad_wts / np.sqrt(np.pi)  # normalize

    # Initial parameter estimates
    p_item = np.clip(X.mean(axis=0), 0.001, 0.999)
    b = -np.log(p_item / (1.0 - p_item))
    a = np.ones(k) * 1.0

    loglik_prev = -np.inf

    for iteration in range(max_iter):
        # E-step: compute posterior weights for each person at each quad point
        # P(X_i | theta_q) = prod_j P_j(theta_q)^x_ij * Q_j(theta_q)^(1-x_ij)
        log_like_quad = np.zeros((n, n_quad))
        for q in range(n_quad):
            for j in range(k):
                Pj = _icc_2pl(quad_pts[q], a[j], b[j])
                Pj = np.clip(Pj, 1e-10, 1.0 - 1e-10)
                log_like_quad[:, q] += X[:, j] * np.log(Pj) + (1.0 - X[:, j]) * np.log(1.0 - Pj)

        # Posterior weights
        log_joint = log_like_quad + np.log(quad_wts)[None, :]
        log_marginal = np.logaddexp.reduce(log_joint, axis=1)
        loglik = np.sum(log_marginal)

        if abs(loglik - loglik_prev) < tol:
            break
        loglik_prev = loglik

        posterior = np.exp(log_joint - log_marginal[:, None])  # n x n_quad

        # Expected counts
        r_bar = posterior.sum(axis=0)  # n_quad, effective N at each quad point
        # Expected number correct at each quad point for each item
        f_bar = np.zeros((k, n_quad))
        for j in range(k):
            f_bar[j, :] = (X[:, j][:, None] * posterior).sum(axis=0)

        # M-step: update a, b for each item
        for j in range(k):

            def _neg_loglik_item(params):
                aj, bj = params
                if aj < 0.01:
                    return 1e12
                Pq = _icc_2pl(quad_pts, aj, bj)
                Pq = np.clip(Pq, 1e-10, 1.0 - 1e-10)
                ll = np.sum(f_bar[j] * np.log(Pq) + (r_bar - f_bar[j]) * np.log(1.0 - Pq))
                return -ll

            res = optimize.minimize(
                _neg_loglik_item,
                x0=[a[j], b[j]],
                method="L-BFGS-B",
                bounds=[(0.01, 5.0), (-6.0, 6.0)],
            )
            if res.success:
                a[j], b[j] = res.x

    # Person ability estimates (EAP)
    log_like_quad = np.zeros((n, n_quad))
    for q in range(n_quad):
        for j in range(k):
            Pj = _icc_2pl(quad_pts[q], a[j], b[j])
            Pj = np.clip(Pj, 1e-10, 1.0 - 1e-10)
            log_like_quad[:, q] += X[:, j] * np.log(Pj) + (1.0 - X[:, j]) * np.log(1.0 - Pj)

    log_joint = log_like_quad + np.log(quad_wts)[None, :]
    log_marginal = np.logaddexp.reduce(log_joint, axis=1)
    posterior = np.exp(log_joint - log_marginal[:, None])

    theta = (posterior * quad_pts[None, :]).sum(axis=1)
    se_theta = np.sqrt((posterior * (quad_pts[None, :] - theta[:, None]) ** 2).sum(axis=1))

    # Fit statistics
    n_params = 2 * k
    aic = -2.0 * loglik + 2.0 * n_params
    bic = -2.0 * loglik + np.log(n) * n_params

    item_params = {}
    for j, name in enumerate(names):
        item_params[name] = {"a": float(a[j]), "b": float(b[j])}

    # Test information on grid
    if theta_grid is None:
        theta_grid = np.linspace(-4, 4, 81)
    info_grid = np.zeros_like(theta_grid)
    for j in range(k):
        Pg = _icc_2pl(theta_grid, a[j], b[j])
        Pg = np.clip(Pg, 1e-10, 1.0 - 1e-10)
        info_grid += a[j] ** 2 * Pg * (1.0 - Pg)

    return IRTResult(
        model="2PL",
        item_params=item_params,
        theta=theta,
        se_theta=se_theta,
        fit={"loglik": float(loglik), "aic": float(aic), "bic": float(bic), "n_iter": iteration + 1, "n": n, "k": k},
        info=info_grid,
    )


twopl = irt2p


def cheatsheet() -> str:
    return "_icc_2pl({}) -> 2-Parameter Logistic IRT model via marginal MLE (EM)."

# morie.fn — function file (hadesllm/morie)
"""3-Parameter Logistic IRT model via marginal MLE (EM)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from morie.fn._containers import IRTResult


def _icc_3pl(theta: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Item characteristic curve for 3PL: P = c + (1-c) / (1 + exp(-a(theta-b)))."""
    logit = np.clip(a * (theta - b), -700, 700)
    return c + (1.0 - c) / (1.0 + np.exp(-logit))


def irt3p(
    data: pd.DataFrame | np.ndarray,
    *,
    n_quad: int = 41,
    max_iter: int = 300,
    tol: float = 1e-5,
    c_prior_alpha: float = 5.0,
    c_prior_beta: float = 17.0,
    theta_grid: np.ndarray | None = None,
) -> IRTResult:
    """Fit a 3-Parameter Logistic IRT model via marginal MLE (EM).

    Adds a lower asymptote (guessing) parameter c to the 2PL model.
    P(theta) = c + (1 - c) / (1 + exp(-a * (theta - b)))

    A Beta(5, 17) prior on c is used to stabilize estimation, following
    Swaminathan and Gifford (1986).

    Parameters
    ----------
    data : DataFrame or ndarray
        Binary item response matrix (n x k), values 0/1.
    n_quad : int
        Number of quadrature points (default 41).
    max_iter : int
        Maximum EM iterations (default 300).
    tol : float
        Convergence tolerance (default 1e-5).
    c_prior_alpha, c_prior_beta : float
        Beta prior hyperparameters on guessing parameter c.
    theta_grid : ndarray, optional
        Grid for test information.

    Returns
    -------
    IRTResult
        model="3PL", item_params={item: {"a", "b", "c"}}.

    References
    ----------
    Birnbaum, A. (1968). Some latent trait models and their use in
    inferring an examinee's ability. In F. M. Lord & M. R. Novick (Eds.),
    Statistical Theories of Mental Test Scores.

    Swaminathan, H. & Gifford, J. A. (1986). Bayesian estimation in the
    three-parameter logistic model. Psychometrika, 51(4), 589-601.
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape
    if k < 2:
        raise ValueError("Need at least 2 items for IRT model.")

    X = np.where(np.isnan(X), 0.0, X)

    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"item_{j}" for j in range(k)]

    # Quadrature
    quad_pts, quad_wts = np.polynomial.hermite.hermgauss(n_quad)
    quad_pts = quad_pts * np.sqrt(2)
    quad_wts = quad_wts / np.sqrt(np.pi)

    # Initial estimates
    p_item = np.clip(X.mean(axis=0), 0.001, 0.999)
    b = -np.log(p_item / (1.0 - p_item))
    a = np.ones(k) * 1.0
    c = np.full(k, 0.2)  # common initial guess for guessing

    loglik_prev = -np.inf

    for iteration in range(max_iter):
        # E-step
        log_like_quad = np.zeros((n, n_quad))
        for q in range(n_quad):
            for j in range(k):
                Pj = _icc_3pl(quad_pts[q], a[j], b[j], c[j])
                Pj = np.clip(Pj, 1e-10, 1.0 - 1e-10)
                log_like_quad[:, q] += X[:, j] * np.log(Pj) + (1.0 - X[:, j]) * np.log(1.0 - Pj)

        log_joint = log_like_quad + np.log(quad_wts)[None, :]
        log_marginal = np.logaddexp.reduce(log_joint, axis=1)
        loglik = np.sum(log_marginal)

        if abs(loglik - loglik_prev) < tol:
            break
        loglik_prev = loglik

        posterior = np.exp(log_joint - log_marginal[:, None])

        r_bar = posterior.sum(axis=0)
        f_bar = np.zeros((k, n_quad))
        for j in range(k):
            f_bar[j, :] = (X[:, j][:, None] * posterior).sum(axis=0)

        # M-step with Beta prior on c
        for j in range(k):

            def _neg_loglik_item(params):
                aj, bj, cj = params
                if aj < 0.01 or cj < 0.0 or cj > 0.5:
                    return 1e12
                Pq = _icc_3pl(quad_pts, aj, bj, cj)
                Pq = np.clip(Pq, 1e-10, 1.0 - 1e-10)
                ll = np.sum(f_bar[j] * np.log(Pq) + (r_bar - f_bar[j]) * np.log(1.0 - Pq))
                # Beta prior on c
                if cj > 1e-10 and cj < 1.0 - 1e-10:
                    ll += (c_prior_alpha - 1) * np.log(cj) + (c_prior_beta - 1) * np.log(1.0 - cj)
                return -ll

            res = optimize.minimize(
                _neg_loglik_item,
                x0=[a[j], b[j], c[j]],
                method="L-BFGS-B",
                bounds=[(0.01, 5.0), (-6.0, 6.0), (0.0, 0.5)],
            )
            if res.success:
                a[j], b[j], c[j] = res.x

    # EAP ability estimates
    log_like_quad = np.zeros((n, n_quad))
    for q in range(n_quad):
        for j in range(k):
            Pj = _icc_3pl(quad_pts[q], a[j], b[j], c[j])
            Pj = np.clip(Pj, 1e-10, 1.0 - 1e-10)
            log_like_quad[:, q] += X[:, j] * np.log(Pj) + (1.0 - X[:, j]) * np.log(1.0 - Pj)

    log_joint = log_like_quad + np.log(quad_wts)[None, :]
    log_marginal = np.logaddexp.reduce(log_joint, axis=1)
    posterior = np.exp(log_joint - log_marginal[:, None])
    theta = (posterior * quad_pts[None, :]).sum(axis=1)
    se_theta = np.sqrt((posterior * (quad_pts[None, :] - theta[:, None]) ** 2).sum(axis=1))

    n_params = 3 * k
    aic = -2.0 * loglik + 2.0 * n_params
    bic = -2.0 * loglik + np.log(n) * n_params

    item_params = {}
    for j, name in enumerate(names):
        item_params[name] = {"a": float(a[j]), "b": float(b[j]), "c": float(c[j])}

    if theta_grid is None:
        theta_grid = np.linspace(-4, 4, 81)
    info_grid = np.zeros_like(theta_grid)
    for j in range(k):
        Pg = _icc_3pl(theta_grid, a[j], b[j], c[j])
        Pg = np.clip(Pg, 1e-10, 1.0 - 1e-10)
        Qg = 1.0 - Pg
        # 3PL information: a^2 * ((Pg - cj)/(1 - cj))^2 * Qg / Pg
        Pstar = (Pg - c[j]) / (1.0 - c[j])
        Pstar = np.clip(Pstar, 1e-10, 1.0)
        info_grid += a[j] ** 2 * Pstar**2 * Qg / Pg

    return IRTResult(
        model="3PL",
        item_params=item_params,
        theta=theta,
        se_theta=se_theta,
        fit={"loglik": float(loglik), "aic": float(aic), "bic": float(bic), "n_iter": iteration + 1, "n": n, "k": k},
        info=info_grid,
    )


threepl = irt3p


def cheatsheet() -> str:
    return "_icc_3pl({}) -> 3-Parameter Logistic IRT model via marginal MLE (EM)."

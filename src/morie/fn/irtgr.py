# morie.fn — function file (hadesllm/morie)
"""Graded Response Model (Samejima) for polytomous IRT."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from morie.fn._containers import IRTResult


def _grm_category_probs(theta: np.ndarray, a: float, thresholds: list[float]) -> np.ndarray:
    """Category response probabilities for one GRM item.

    Returns array of shape (len(theta), n_categories) where n_categories = len(thresholds) + 1.
    """
    n_t = len(theta)
    m = len(thresholds)  # number of thresholds = n_categories - 1

    # Cumulative probabilities: P*(theta, b_k) = 1/(1+exp(-a(theta-b_k)))
    # P*(theta, b_0) = 1 (below lowest), P*(theta, b_m+1) = 0 (above highest)
    cum = np.ones((n_t, m + 2))
    cum[:, -1] = 0.0
    for kk in range(m):
        logit = np.clip(a * (theta - thresholds[kk]), -700, 700)
        cum[:, kk + 1] = 1.0 / (1.0 + np.exp(-logit))

    # Category probabilities: P(k) = P*(k) - P*(k+1)
    probs = np.diff(-cum, axis=1)  # cum[:, :-1] - cum[:, 1:]
    # Actually: probs[:, k] = cum[:, k] - cum[:, k+1]
    probs2 = np.zeros((n_t, m + 1))
    for kk in range(m + 1):
        probs2[:, kk] = cum[:, kk] - cum[:, kk + 1]
    return np.clip(probs2, 1e-10, 1.0)


def irtgr(
    data: pd.DataFrame | np.ndarray,
    *,
    n_quad: int = 41,
    max_iter: int = 200,
    tol: float = 1e-5,
    theta_grid: np.ndarray | None = None,
) -> IRTResult:
    """Fit Samejima's Graded Response Model for polytomous items.

    Suitable for ordered-category (Likert) data.  Estimates item
    discrimination (a) and category threshold (b_k) parameters.

    Parameters
    ----------
    data : DataFrame or ndarray
        Item response matrix (n x k).  Responses are integers starting at 0.
        E.g. for 5-point Likert coded 1-5, subtract 1 first or pass as-is
        (the function auto-detects minimum and shifts to 0-based).
    n_quad : int
        Number of quadrature points (default 41).
    max_iter : int
        Maximum EM iterations (default 200).
    tol : float
        Convergence tolerance (default 1e-5).
    theta_grid : ndarray, optional
        Grid for test information.

    Returns
    -------
    IRTResult
        model="GRM", item_params={item: {"a": float, "thresholds": list[float]}}.

    References
    ----------
    Samejima, F. (1969). Estimation of latent ability using a response
    pattern of graded scores. Psychometrika Monograph Supplement, No. 17.
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape
    if k < 2:
        raise ValueError("Need at least 2 items for GRM.")

    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"item_{j}" for j in range(k)]

    # Shift to 0-based per item
    X_int = np.copy(X)
    n_cats = np.zeros(k, dtype=int)
    for j in range(k):
        col = X_int[:, j]
        valid = col[np.isfinite(col)]
        if len(valid) == 0:
            n_cats[j] = 2
            continue
        min_val = int(np.nanmin(valid))
        X_int[:, j] = col - min_val
        n_cats[j] = int(np.nanmax(valid) - min_val) + 1

    X_int = np.where(np.isnan(X_int), 0, X_int).astype(int)

    # Quadrature
    quad_pts, quad_wts = np.polynomial.hermite.hermgauss(n_quad)
    quad_pts = quad_pts * np.sqrt(2)
    quad_wts = quad_wts / np.sqrt(np.pi)

    # Initial estimates
    a = np.ones(k) * 1.0
    thresholds = []
    for j in range(k):
        m = n_cats[j] - 1  # number of thresholds
        if m < 1:
            thresholds.append([0.0])
        else:
            thresholds.append(list(np.linspace(-1.5, 1.5, m)))

    loglik_prev = -np.inf

    for iteration in range(max_iter):
        # E-step
        log_like_quad = np.zeros((n, n_quad))
        for q in range(n_quad):
            th_q = np.array([quad_pts[q]])
            for j in range(k):
                probs = _grm_category_probs(th_q, a[j], thresholds[j])  # (1, n_cats)
                resp = X_int[:, j]
                resp_clipped = np.clip(resp, 0, probs.shape[1] - 1)
                log_like_quad[:, q] += np.log(probs[0, resp_clipped])

        log_joint = log_like_quad + np.log(quad_wts)[None, :]
        log_marginal = np.logaddexp.reduce(log_joint, axis=1)
        loglik = np.sum(log_marginal)

        if abs(loglik - loglik_prev) < tol:
            break
        loglik_prev = loglik

        posterior = np.exp(log_joint - log_marginal[:, None])

        # M-step: optimize each item separately
        for j in range(k):
            m = n_cats[j] - 1
            if m < 1:
                continue

            def _neg_ll_item(params, _j=j, _m=m):
                aj = params[0]
                bj = list(params[1:])
                if aj < 0.01:
                    return 1e12
                # Ensure thresholds are ordered
                for kk in range(len(bj) - 1):
                    if bj[kk] >= bj[kk + 1]:
                        return 1e12

                ll = 0.0
                for q in range(n_quad):
                    th_q = np.array([quad_pts[q]])
                    probs = _grm_category_probs(th_q, aj, bj)
                    resp = X_int[:, _j]
                    resp_clipped = np.clip(resp, 0, probs.shape[1] - 1)
                    ll += np.sum(posterior[:, q] * np.log(probs[0, resp_clipped]))
                return -ll

            x0 = [a[j]] + thresholds[j]
            bounds = [(0.01, 5.0)] + [(-6.0, 6.0)] * m

            res = optimize.minimize(_neg_ll_item, x0=x0, method="L-BFGS-B", bounds=bounds)
            if res.success:
                a[j] = res.x[0]
                thresholds[j] = sorted(res.x[1:].tolist())

    # EAP
    log_like_quad = np.zeros((n, n_quad))
    for q in range(n_quad):
        th_q = np.array([quad_pts[q]])
        for j in range(k):
            probs = _grm_category_probs(th_q, a[j], thresholds[j])
            resp_clipped = np.clip(X_int[:, j], 0, probs.shape[1] - 1)
            log_like_quad[:, q] += np.log(probs[0, resp_clipped])

    log_joint = log_like_quad + np.log(quad_wts)[None, :]
    log_marginal = np.logaddexp.reduce(log_joint, axis=1)
    posterior = np.exp(log_joint - log_marginal[:, None])
    theta = (posterior * quad_pts[None, :]).sum(axis=1)
    se_theta = np.sqrt((posterior * (quad_pts[None, :] - theta[:, None]) ** 2).sum(axis=1))

    n_params = sum(1 + (n_cats[j] - 1) for j in range(k))
    aic = -2.0 * loglik + 2.0 * n_params
    bic = -2.0 * loglik + np.log(n) * n_params

    item_params = {}
    for j, name in enumerate(names):
        item_params[name] = {
            "a": float(a[j]),
            "thresholds": [float(t) for t in thresholds[j]],
        }

    if theta_grid is None:
        theta_grid = np.linspace(-4, 4, 81)
    info_grid = np.zeros_like(theta_grid)
    for j in range(k):
        m = n_cats[j] - 1
        th_arr = theta_grid
        probs = _grm_category_probs(th_arr, a[j], thresholds[j])
        # GRM information: sum over categories of (P'_k)^2 / P_k
        # P'_k for GRM involves cumulative prob derivatives
        for kk in range(probs.shape[1]):
            # Numerical derivative of P_k w.r.t. theta
            eps = 1e-5
            probs_plus = _grm_category_probs(th_arr + eps, a[j], thresholds[j])
            probs_minus = _grm_category_probs(th_arr - eps, a[j], thresholds[j])
            dP = (probs_plus[:, kk] - probs_minus[:, kk]) / (2 * eps)
            info_grid += dP**2 / np.clip(probs[:, kk], 1e-10, None)

    return IRTResult(
        model="GRM",
        item_params=item_params,
        theta=theta,
        se_theta=se_theta,
        fit={"loglik": float(loglik), "aic": float(aic), "bic": float(bic), "n_iter": iteration + 1, "n": n, "k": k},
        info=info_grid,
    )


grm = irtgr


def cheatsheet() -> str:
    return "_grm_category_probs({}) -> Graded Response Model (Samejima) for polytomous IRT."

# moirais.fn — function file (hadesllm/moirais)
"""Partial Credit Model (Masters) for polytomous IRT."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from moirais.fn._containers import IRTResult


def _pcm_category_probs(theta: float, steps: list[float]) -> np.ndarray:
    """Category probabilities for one PCM item at a single theta.

    P(X=k | theta) = exp(sum_{v=0}^{k} (theta - d_v)) / sum_c exp(sum_{v=0}^{c} (theta - d_v))

    where d_0 = 0 by convention.
    Returns array of shape (n_categories,).
    """
    m = len(steps)  # number of step parameters = n_categories - 1
    n_cat = m + 1

    # Numerators: exp(cumulative sum of (theta - d_v))
    log_num = np.zeros(n_cat)
    cumsum = 0.0
    for kk in range(1, n_cat):
        cumsum += theta - steps[kk - 1]
        log_num[kk] = cumsum

    # Normalize in log space for stability
    log_max = np.max(log_num)
    log_num_shifted = log_num - log_max
    denom = np.sum(np.exp(log_num_shifted))
    probs = np.exp(log_num_shifted) / denom
    return np.clip(probs, 1e-10, 1.0)


def irtpc(
    data: pd.DataFrame | np.ndarray,
    *,
    n_quad: int = 41,
    max_iter: int = 200,
    tol: float = 1e-5,
    theta_grid: np.ndarray | None = None,
) -> IRTResult:
    """Fit the Partial Credit Model (Masters, 1982).

    An extension of the Rasch model for ordered polytomous data.
    Estimates item-specific step difficulty parameters with discrimination
    fixed at a=1.  Each item can have a different number of categories.

    Parameters
    ----------
    data : DataFrame or ndarray
        Item response matrix (n x k).  Responses are integers.
        Automatically shifted to 0-based per item.
    n_quad : int
        Quadrature points (default 41).
    max_iter : int
        Maximum EM iterations (default 200).
    tol : float
        Convergence tolerance (default 1e-5).
    theta_grid : ndarray, optional
        Grid for test information.

    Returns
    -------
    IRTResult
        model="PCM", item_params={item: {"steps": list[float]}}.

    References
    ----------
    Masters, G. N. (1982). A Rasch model for partial credit scoring.
    Psychometrika, 47(2), 149-174.
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape
    if k < 2:
        raise ValueError("Need at least 2 items for PCM.")

    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"item_{j}" for j in range(k)]

    # Shift to 0-based
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

    # Initial step parameters
    steps_list: list[list[float]] = []
    for j in range(k):
        m = n_cats[j] - 1
        if m < 1:
            steps_list.append([0.0])
        else:
            steps_list.append(list(np.linspace(-1.0, 1.0, m)))

    loglik_prev = -np.inf

    for iteration in range(max_iter):
        # E-step
        log_like_quad = np.zeros((n, n_quad))
        for q in range(n_quad):
            for j in range(k):
                probs = _pcm_category_probs(quad_pts[q], steps_list[j])
                resp = np.clip(X_int[:, j], 0, len(probs) - 1)
                log_like_quad[:, q] += np.log(probs[resp])

        log_joint = log_like_quad + np.log(quad_wts)[None, :]
        log_marginal = np.logaddexp.reduce(log_joint, axis=1)
        loglik = np.sum(log_marginal)

        if abs(loglik - loglik_prev) < tol:
            break
        loglik_prev = loglik

        posterior = np.exp(log_joint - log_marginal[:, None])

        # M-step
        for j in range(k):
            m = n_cats[j] - 1
            if m < 1:
                continue

            def _neg_ll(params, _j=j):
                steps = list(params)
                ll = 0.0
                for q in range(n_quad):
                    probs = _pcm_category_probs(quad_pts[q], steps)
                    resp = np.clip(X_int[:, _j], 0, len(probs) - 1)
                    ll += np.sum(posterior[:, q] * np.log(probs[resp]))
                return -ll

            res = optimize.minimize(
                _neg_ll,
                x0=steps_list[j],
                method="L-BFGS-B",
                bounds=[(-6.0, 6.0)] * m,
            )
            if res.success:
                steps_list[j] = res.x.tolist()

    # EAP
    log_like_quad = np.zeros((n, n_quad))
    for q in range(n_quad):
        for j in range(k):
            probs = _pcm_category_probs(quad_pts[q], steps_list[j])
            resp = np.clip(X_int[:, j], 0, len(probs) - 1)
            log_like_quad[:, q] += np.log(probs[resp])

    log_joint = log_like_quad + np.log(quad_wts)[None, :]
    log_marginal = np.logaddexp.reduce(log_joint, axis=1)
    posterior = np.exp(log_joint - log_marginal[:, None])
    theta = (posterior * quad_pts[None, :]).sum(axis=1)
    se_theta = np.sqrt((posterior * (quad_pts[None, :] - theta[:, None]) ** 2).sum(axis=1))

    n_params = sum(max(n_cats[j] - 1, 1) for j in range(k))
    aic = -2.0 * loglik + 2.0 * n_params
    bic = -2.0 * loglik + np.log(n) * n_params

    item_params = {}
    for j, name in enumerate(names):
        item_params[name] = {"steps": [float(s) for s in steps_list[j]]}

    # Test information via numerical differentiation
    if theta_grid is None:
        theta_grid = np.linspace(-4, 4, 81)
    info_grid = np.zeros_like(theta_grid)
    eps = 1e-5
    for j in range(k):
        for idx, th in enumerate(theta_grid):
            probs = _pcm_category_probs(th, steps_list[j])
            probs_p = _pcm_category_probs(th + eps, steps_list[j])
            probs_m = _pcm_category_probs(th - eps, steps_list[j])
            dP = (probs_p - probs_m) / (2 * eps)
            info_grid[idx] += np.sum(dP**2 / probs)

    return IRTResult(
        model="PCM",
        item_params=item_params,
        theta=theta,
        se_theta=se_theta,
        fit={"loglik": float(loglik), "aic": float(aic), "bic": float(bic), "n_iter": iteration + 1, "n": n, "k": k},
        info=info_grid,
    )


pcm = irtpc


def cheatsheet() -> str:
    return "_pcm_category_probs({}) -> Partial Credit Model (Masters) for polytomous IRT."

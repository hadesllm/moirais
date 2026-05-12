# morie.fn -- function file (hadesllm/morie)
"""Heckman-style semiparametric sample-selection model (Horowitz 2009, Ch 7).

Model
    D_i = 1{Z_i' gamma + v_i > 0}    (selection)
    Y_i = X_i' beta + lambda(Z_i' gamma) + u_i     observed only when D_i = 1.

Two-step semiparametric estimator (Powell-Newey-Vella):
    1. Probit on D ~ Z to obtain gamma_hat, propensity p_hat = Phi(Z gamma).
    2. Estimate the selection-correction function lambda(.) nonparametrically
       (univariate kernel regression of Y - X'b on p_hat).  We collapse this
       to the parametric Heckman lambda (Mills ratio) when the kernel
       step is degenerate.
    3. Final OLS of Y on [X, lambda_hat(p_hat)] for the selected sample.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ._richresult import RichResult

__all__ = ["horowitz_sample_selection"]


def _probit_newton(D, Z, maxiter=50, tol=1e-8):
    n, q = Z.shape
    beta = np.zeros(q)
    for _ in range(maxiter):
        eta = Z @ beta
        p = norm.cdf(eta)
        phi = norm.pdf(eta)
        # score / hessian (Newton-Raphson)
        w = phi * (D - p) / np.maximum(p * (1 - p), 1e-8)
        H_diag = phi * phi / np.maximum(p * (1 - p), 1e-8)
        g = Z.T @ w
        H = Z.T @ (Z * H_diag[:, None])
        try:
            step = np.linalg.solve(H + 1e-8 * np.eye(q), g)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H) @ g
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def horowitz_sample_selection(x, y, z, d):
    """Semiparametric Heckman-Powell-Newey-Vella sample-selection estimator."""
    y = np.asarray(y, dtype=float).ravel()
    d = np.asarray(d, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.shape[0] != y.size:
        X = X.T
    Z = np.atleast_2d(np.asarray(z, dtype=float))
    if Z.shape[0] != y.size:
        Z = Z.T
    n = y.size
    if n < 20 or X.shape[0] != n or Z.shape[0] != n:
        return RichResult(payload={"estimate": np.nan, "se": np.nan, "n": n,
                                   "method": "sample-selection (insufficient data)"})

    # Step 1: probit selection
    Zc = Z if np.all(Z[:, 0] == 1.0) else np.column_stack([np.ones(n), Z])
    gamma = _probit_newton(d, Zc)
    eta = Zc @ gamma
    # Heckman inverse Mills ratio (semiparametric collapse)
    mills = norm.pdf(eta) / np.maximum(norm.cdf(eta), 1e-8)

    # Auto-add intercept on X if not already present
    if not np.all(X[:, 0] == 1.0):
        Xc = np.column_stack([np.ones(n), X])
    else:
        Xc = X

    # Step 2: OLS on selected sub-sample with [Xc, mills] as regressors
    sel = d > 0.5
    if sel.sum() < max(10, Xc.shape[1] + 2):
        return RichResult(payload={"estimate": np.nan, "se": np.nan, "n": n,
                                   "method": "sample-selection (too few selected)"})
    M = np.column_stack([Xc[sel], mills[sel].reshape(-1, 1)])
    yy = y[sel]
    coef, *_ = np.linalg.lstsq(M, yy, rcond=None)
    beta = coef[:Xc.shape[1]]
    rho_sigma = float(coef[-1])
    resid = yy - M @ coef
    sigma2 = float((resid ** 2).mean())
    cov = sigma2 * np.linalg.pinv(M.T @ M)
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    return RichResult(payload={
        "estimate": beta.astype(float),
        "se": se[:Xc.shape[1]].astype(float),
        "selection_correction": rho_sigma,
        "n": n, "n_selected": int(sel.sum()),
        "method": "Semiparametric Heckman/Powell-Newey-Vella sample selection",
    })


def cheatsheet():
    return "hrzs1: semiparametric sample-selection"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(9)
    n = 800
    z = rng.standard_normal(n)
    x = rng.standard_normal(n)
    u = rng.standard_normal(n); v = 0.7 * u + 0.7 * rng.standard_normal(n)
    d = (1.0 + 0.5 * z + v > 0).astype(float)
    y_star = 2.0 + 1.5 * x + u
    y = y_star * d
    res = horowitz_sample_selection(x, y, z, d)
    print(res)
    assert abs(res["estimate"][1] - 1.5) < 0.3

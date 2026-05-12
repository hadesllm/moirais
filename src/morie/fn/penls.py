# morie.fn — function file (hadesllm/morie)
"""Penalized regression (ridge / LASSO / elastic net) for markers."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["penalized_regression"]


def penalized_regression(x, y, alpha: float = 0.5, lam: float = 1.0,
                         max_iter: int = 1000, tol: float = 1e-6):
    """Elastic-net regression via coordinate descent (NumPy fallback).

    Objective::

        min_b 1/(2n) ||y - X b||^2 + lam*(alpha*||b||_1 + (1-alpha)/2*||b||_2^2)

    Parameters
    ----------
    x : array-like (n, p)
    y : array-like (n,)
    alpha : float in [0,1]. 0 = ridge, 1 = LASSO, in-between = elastic net.
    lam : float >= 0. Penalty strength.
    max_iter, tol : convergence controls.

    Returns
    -------
    RichResult with payload keys estimate (mean |beta|), beta, intercept,
    se, alpha, lam, n_iter, n, p, method.

    References
    ----------
    Zou, H., & Hastie, T. (2005). Regularization and variable selection
        via the elastic net. JRSS-B, 67(2), 301-320.
    Friedman, Hastie & Tibshirani (2010). Regularization paths for
        generalized linear models via coordinate descent. JSS, 33(1), 1-22.
    Montesinos Lopez et al. (2022), Ch. 6.
    """
    X = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    # Centre
    y_mean = float(np.mean(y)); yc = y - y_mean
    x_mean = X.mean(axis=0)
    x_sd = X.std(axis=0); x_sd = np.where(x_sd > 0, x_sd, 1.0)
    Xs = (X - x_mean) / x_sd
    # Coordinate descent on standardised X
    beta = np.zeros(p)
    XtX_diag = np.einsum("ij,ij->j", Xs, Xs) / n
    r = yc - Xs @ beta
    soft = lam * alpha
    ridge_term = lam * (1 - alpha)
    n_iter_done = max_iter
    for it in range(max_iter):
        max_change = 0.0
        for j in range(p):
            r_j = r + Xs[:, j] * beta[j]
            z = float(Xs[:, j] @ r_j) / n
            # Soft-thresholding
            if z > soft:
                new = (z - soft) / (XtX_diag[j] + ridge_term)
            elif z < -soft:
                new = (z + soft) / (XtX_diag[j] + ridge_term)
            else:
                new = 0.0
            change = new - beta[j]
            if abs(change) > max_change:
                max_change = abs(change)
            beta[j] = new
            r = r_j - Xs[:, j] * new
        if max_change < tol:
            n_iter_done = it + 1
            break
    # Un-standardise
    beta_orig = beta / x_sd
    intercept = y_mean - float(x_mean @ beta_orig)
    y_hat = X @ beta_orig + intercept
    resid = y - y_hat
    se = float(np.sqrt(np.sum(resid ** 2) / max(n - p, 1)))
    return RichResult(
        title=f"Penalised regression (alpha={alpha}, lam={lam})",
        summary_lines=[
            ("n", n),
            ("p", p),
            ("alpha", alpha),
            ("lambda", lam),
            ("n_iter", n_iter_done),
            ("non-zero coefs", int(np.sum(np.abs(beta_orig) > 1e-8))),
            ("residual SE", se),
        ],
        payload={
            "estimate": float(np.mean(np.abs(beta_orig))),
            "beta": beta_orig,
            "intercept": intercept,
            "y_hat": y_hat,
            "se": se,
            "alpha": alpha,
            "lam": lam,
            "n_iter": n_iter_done,
            "n": n,
            "p": p,
            "method": "Elastic-net coordinate descent",
        },
    )


def cheatsheet():
    return "penls: Penalised regression (ridge/LASSO/elastic net)"


# CANONICAL TEST
# np.random.seed(10); X = np.random.randn(30, 4); beta_true = np.array([1,0,-1,0])
# y = X @ beta_true + 0.1*np.random.randn(30)
# r = penalized_regression(X, y, alpha=1.0, lam=0.05); r.beta[1]≈0, r.beta[3]≈0.

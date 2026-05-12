# morie.fn — function file (hadesllm/morie)
"""Bayesian ridge regression (RR-BLUP) for marker effects."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["bayesian_ridge_regression"]


def bayesian_ridge_regression(x, y, lam: float | None = None):
    """Bayesian ridge regression with a Normal-Inverse-Gamma prior.

    Model::

        y = X*beta + e,  beta_j ~ N(0, sigma_b^2),  sigma_b^2 ~ IG(a,b)

    The posterior mode of beta given variance components is the ridge
    estimator::

        beta_hat = (X'X + lam*I)^{-1} X'y,  lam = sigma_e^2 / sigma_b^2

    This is the RR-BLUP closed-form solution used throughout Montesinos
    Lopez Ch 4. If `lam` is not supplied we use a single-pass empirical
    Bayes update: lam = var(y) / mean(X^2) / p as a heuristic starting
    point (cf. Endelman 2011 rrBLUP).

    Parameters
    ----------
    x : array-like (n,p)  — marker / design matrix
    y : array-like (n,)
    lam : float, optional. Ridge parameter sigma_e^2 / sigma_b^2.

    Returns
    -------
    RichResult with payload keys estimate (mean abs beta), beta, se,
    lambda, n, p, method.

    References
    ----------
    Montesinos Lopez et al. (2022), Ch. 4.
    """
    X = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    # Centre y and X so we don't need a separate intercept
    y_mean = float(np.mean(y))
    yc = y - y_mean
    x_mean = X.mean(axis=0)
    Xc = X - x_mean
    if lam is None:
        var_y = float(np.var(yc, ddof=1)) if n > 1 else 1.0
        # Endelman rrBLUP-style starting lambda: heritability 0.5
        # var_g = h2 var_y; sigma_b^2 = var_g / p; lam = var_e/sigma_b^2
        h2 = 0.5
        var_e = (1.0 - h2) * var_y
        sigma_b2 = (h2 * var_y) / max(p, 1)
        lam = float(var_e / sigma_b2) if sigma_b2 > 0 else 1.0
    XtX = Xc.T @ Xc
    A = XtX + lam * np.eye(p)
    beta = np.linalg.solve(A, Xc.T @ yc)
    intercept = y_mean
    y_hat = Xc @ beta + intercept
    resid = y - y_hat
    sigma2 = float(np.sum(resid ** 2) / max(n - 1, 1))
    # SE from posterior covariance: sigma2 * (X'X + lam I)^{-1}
    cov_beta = sigma2 * np.linalg.inv(A)
    se = np.sqrt(np.clip(np.diag(cov_beta), 0, None))
    return RichResult(
        title="Bayesian Ridge Regression (RR-BLUP)",
        summary_lines=[
            ("n", n),
            ("p", p),
            ("lambda", lam),
            ("residual sigma^2", sigma2),
            ("mean |beta|", float(np.mean(np.abs(beta)))),
        ],
        payload={
            "estimate": float(np.mean(np.abs(beta))),
            "beta": beta,
            "intercept": intercept,
            "se": float(np.mean(se)),
            "beta_se": se,
            "lam": float(lam),
            "n": n,
            "p": p,
            "method": "Bayesian ridge regression (closed-form posterior mode)",
        },
    )


def cheatsheet():
    return "brreg: Bayesian ridge regression for marker effects"


# CANONICAL TEST
# np.random.seed(2); X = np.random.randn(20, 5); beta_true = np.array([1,-1,0.5,0,0])
# y = X @ beta_true + 0.1*np.random.randn(20)
# r = bayesian_ridge_regression(X, y); r.beta ≈ beta_true with small bias.

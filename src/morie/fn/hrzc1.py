# morie.fn — function file (hadesllm/morie)
"""Powell (1984) censored least-absolute-deviations (Horowitz 2009, Ch 6).

Model:   Y* = X'beta + u,   Y = max(c, Y*)

Powell CLAD estimator:
    beta = argmin sum_i |Y_i - max(c, X_i'b)|

iterative reweighting via the equivalent quantile-regression on the
sub-sample with ``X_i'b > c``.
"""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["horowitz_censored_regression"]


def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _qreg_irls(X, y, tau=0.5, maxiter=50, tol=1e-6):
    """Quantile regression via IRLS (median when tau=0.5)."""
    n, p = X.shape
    beta = _ols(X, y)
    for _ in range(maxiter):
        r = y - X @ beta
        # rho-prime weights for tau=0.5 -> sign(r)/2; equivalent IRLS
        # w = 1 / max(|r|, eps)
        w = 1.0 / np.maximum(np.abs(r), 1e-4)
        WX = X * w[:, None]
        XtWX = X.T @ WX
        try:
            new = np.linalg.solve(XtWX, WX.T @ y)
        except np.linalg.LinAlgError:
            new = np.linalg.pinv(XtWX) @ (WX.T @ y)
        if np.max(np.abs(new - beta)) < tol:
            beta = new; break
        beta = new
    return beta


def horowitz_censored_regression(x, y, censor=0.0):
    """Powell CLAD estimator for left-censored regression."""
    y = np.asarray(y, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.shape[0] != y.size:
        X = X.T
    n, p = X.shape
    if n < max(10, 2 * p):
        return RichResult(payload={"estimate": np.full(p, np.nan),
                                   "se": np.full(p, np.nan), "n": n,
                                   "method": "CLAD (insufficient data)"})
    c = float(censor)
    # Initialise on uncensored sub-sample
    keep = y > c
    if keep.sum() < max(5, p + 1):
        return RichResult(payload={"estimate": np.full(p, np.nan),
                                   "se": np.full(p, np.nan), "n": n,
                                   "method": "CLAD (too few uncensored obs)"})
    beta = _qreg_irls(X[keep], y[keep])
    # Iterate: only obs with X_i'beta > c contribute
    for _ in range(30):
        active = (X @ beta) > c
        if active.sum() < max(5, p + 1):
            break
        new = _qreg_irls(X[active], y[active])
        if np.max(np.abs(new - beta)) < 1e-5:
            beta = new; break
        beta = new

    # Powell SE: kernel density of residuals at zero
    r = y - X @ beta
    active = (X @ beta) > c
    if active.sum() < max(5, p + 1):
        se = np.full(p, np.nan)
    else:
        Xa = X[active]; ra = r[active]
        h = max(1.06 * np.std(ra, ddof=1) * len(ra) ** (-1/5), 1e-4)
        f0 = float(np.mean(np.exp(-0.5 * (ra / h) ** 2) / (h * np.sqrt(2 * np.pi))))
        A = Xa.T @ Xa * f0
        try:
            cov = 0.25 * np.linalg.pinv(A) @ (Xa.T @ Xa) @ np.linalg.pinv(A)
        except np.linalg.LinAlgError:
            cov = np.full((p, p), np.nan)
        se = np.sqrt(np.maximum(np.diag(cov), 0))

    return RichResult(payload={
        "estimate": beta.astype(float) if beta.size > 1 else float(beta[0]),
        "se": se.astype(float) if se.size > 1 else float(se[0]),
        "n": n, "n_uncensored": int(active.sum()), "censor": c,
        "method": "Powell (1984) censored LAD (CLAD)",
    })


def cheatsheet():
    return "hrzc1: Powell censored LAD"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(8)
    n = 500
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    beta_true = np.array([1.0, 2.0])
    y_star = X @ beta_true + rng.standard_normal(n)
    y = np.maximum(0.0, y_star)
    res = horowitz_censored_regression(X, y, censor=0.0)
    print(res)
    err = np.linalg.norm(res["estimate"] - beta_true)
    assert err < 0.5, f"err={err}"

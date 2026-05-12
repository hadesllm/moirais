# morie.fn -- function file (hadesllm/morie)
"""Wild bootstrap (Wu 1986; Mammen 1993; Horowitz 2009, Ch 13).

Given fitted residuals ``r_i = Y_i - f(X_i, beta_hat)``, the wild
bootstrap draws

    e_i* = r_i * v_i,   v_i in {-1, +1} with prob 1/2  (Rademacher)

and rebuilds Y_i* = X_i' beta_hat + e_i*.  This estimator returns the
bootstrap distribution of OLS beta (or a user-supplied statistic) and
its standard error.
"""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["horowitz_wild_bootstrap"]


def horowitz_wild_bootstrap(x, y, residuals=None, B=500, statistic="ols",
                             seed=0):
    """Rademacher wild bootstrap for OLS coefficients.

    Parameters
    ----------
    x : array-like, (n, k) or (n,)
    y : array-like, (n,)
    residuals : array-like, optional
        Pre-computed residuals; defaults to OLS residuals of y on x.
    B : int, default 500
        Number of bootstrap replications.
    statistic : {"ols"} or callable
    seed : int
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.shape[0] != y.size:
        X = X.T
    n, p = X.shape
    if n < max(10, 2 * p):
        return RichResult(payload={"estimate": np.nan, "se": np.nan, "n": n,
                                   "method": "wild-bootstrap (insufficient data)"})
    if residuals is None:
        beta0, *_ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ beta0
    else:
        beta0, *_ = np.linalg.lstsq(X, y, rcond=None)
        residuals = np.asarray(residuals, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    boot = np.zeros((B, p))
    XtX_inv = np.linalg.pinv(X.T @ X)
    for b in range(B):
        v = rng.choice([-1.0, 1.0], size=n)        # Rademacher
        e_star = residuals * v
        y_star = X @ beta0 + e_star
        boot[b] = XtX_inv @ (X.T @ y_star)
    mean = boot.mean(axis=0)
    se = boot.std(axis=0, ddof=1)
    ci_lo = np.percentile(boot, 2.5, axis=0)
    ci_hi = np.percentile(boot, 97.5, axis=0)
    if p == 1:
        return RichResult(payload={
            "estimate": float(beta0[0]),
            "se": float(se[0]),
            "ci_lower": float(ci_lo[0]),
            "ci_upper": float(ci_hi[0]),
            "boot_mean": float(mean[0]),
            "B": B, "n": n,
            "method": "Rademacher wild bootstrap (Mammen 1993)",
        })
    return RichResult(payload={
        "estimate": beta0.astype(float),
        "se": se.astype(float),
        "ci_lower": ci_lo.astype(float),
        "ci_upper": ci_hi.astype(float),
        "boot_mean": mean.astype(float),
        "B": B, "n": n,
        "method": "Rademacher wild bootstrap (Mammen 1993)",
    })


def cheatsheet():
    return "hrzw1: wild bootstrap (Rademacher)"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(17)
    n = 300
    x = rng.standard_normal((n, 2))
    y = x @ np.array([1.0, -2.0]) + (np.abs(x[:, 0]) + 0.1) * rng.standard_normal(n)
    res = horowitz_wild_bootstrap(x, y, B=200)
    print(res)
    assert abs(res["estimate"][0] - 1.0) < 0.3

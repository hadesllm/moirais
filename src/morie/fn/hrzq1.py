# morie.fn — function file (hadesllm/morie)
"""Semiparametric quantile regression (Horowitz 2009, Ch 10).

    Q_tau(Y | X) = X' beta(tau)

Implementation: classical Koenker-Bassett linear quantile regression via
iteratively reweighted least squares.  Hall-Sheather rule of thumb for
the bandwidth used in the asymptotic SE.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ._richresult import RichResult

__all__ = ["horowitz_quantile_regression"]


def _qreg_irls(X, y, tau, maxiter=100, tol=1e-7):
    """Quantile-regression IRLS (Hunter-Lange 2000)."""
    n, p = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(maxiter):
        r = y - X @ beta
        w = np.where(r > 0, tau / np.maximum(r, 1e-6),
                     (1 - tau) / np.maximum(-r, 1e-6))
        WX = X * w[:, None]
        try:
            new = np.linalg.solve(X.T @ WX, WX.T @ y)
        except np.linalg.LinAlgError:
            new = np.linalg.pinv(X.T @ WX) @ (WX.T @ y)
        if np.max(np.abs(new - beta)) < tol:
            beta = new; break
        beta = new
    return beta


def horowitz_quantile_regression(x, y, tau=0.5):
    """Linear quantile regression at quantile ``tau``.

    Parameters
    ----------
    x : array-like
    y : array-like
    tau : float, default 0.5
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.shape[0] != y.size:
        X = X.T
    n, p = X.shape
    tau = float(tau)
    if n < max(10, 2 * p) or not (0.0 < tau < 1.0):
        return RichResult(payload={"estimate": np.full(p, np.nan),
                                   "se": np.full(p, np.nan), "n": n, "tau": tau,
                                   "method": "QReg (insufficient data or invalid tau)"})
    # Add intercept if user didn't
    if not np.allclose(X[:, 0], 1.0):
        Xp = np.column_stack([np.ones(n), X]); has_intercept = False
    else:
        Xp = X; has_intercept = True
    beta = _qreg_irls(Xp, y, tau)
    resid = y - Xp @ beta
    # Hall-Sheather bandwidth + Powell SE
    h = (norm.ppf(1 - 0.05) ** (2/3)) * \
        ((1.5 * norm.pdf(norm.ppf(tau)) ** 2) /
         (2 * norm.ppf(tau) ** 2 + 1)) ** (1/3) * n ** (-1/3)
    h = max(h, 1e-3)
    f0 = float(np.mean(np.abs(resid) < h) / (2 * h))
    if f0 < 1e-6:
        f0 = 1e-6
    cov = (tau * (1 - tau) / (f0 ** 2)) * np.linalg.pinv(Xp.T @ Xp)
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    if not has_intercept:
        beta_out = beta[1:] if Xp.shape[1] > 1 else beta
        se_out = se[1:] if Xp.shape[1] > 1 else se
    else:
        beta_out = beta; se_out = se
    return RichResult(payload={
        "estimate": beta_out.astype(float) if beta_out.size > 1 else float(beta_out[0]),
        "se": se_out.astype(float) if se_out.size > 1 else float(se_out[0]),
        "intercept": float(beta[0]) if not has_intercept else None,
        "tau": tau, "n": n,
        "method": "Koenker-Bassett quantile regression (IRLS)",
    })


def cheatsheet():
    return "hrzq1: quantile regression"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(13)
    n = 800
    x = rng.standard_normal(n)
    y = 1 + 2 * x + rng.standard_normal(n)
    res = horowitz_quantile_regression(x, y, tau=0.5)
    print(res)
    assert abs(res["estimate"] - 2.0) < 0.2

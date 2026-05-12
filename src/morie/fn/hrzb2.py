# morie.fn -- function file (hadesllm/morie)
"""Horowitz (1992) smoothed maximum-score estimator (Horowitz 2009, Ch 5).

    beta_hat = argmax_{||b||=1} (1/n) sum_i (2 Y_i - 1) * Phi(X_i' b / h)

with Phi the standard normal CDF acting as the smoothing kernel; n^(2/5)
convergence rate and normal-theory SEs.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from ._richresult import RichResult

__all__ = ["horowitz_smoothed_maximum_score"]


def _silverman(x):
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2:
        return 1.0
    s = float(np.std(x, ddof=1))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    sigma = min(s, iqr / 1.349) if iqr > 0 else s
    if sigma <= 0:
        sigma = max(s, 1e-6)
    return 1.06 * sigma * n ** (-1.0 / 5.0)


def horowitz_smoothed_maximum_score(x, y, bandwidth=None):
    """Horowitz smoothed maximum-score estimator."""
    y = np.asarray(y, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.shape[0] != y.size:
        X = X.T
    n, p = X.shape
    if n < max(10, 2 * p):
        return RichResult(payload={"estimate": np.full(p, np.nan),
                                   "se": np.full(p, np.nan), "n": n,
                                   "method": "smoothed-max-score (insufficient data)"})
    y_signed = 2 * y - 1
    h = float(bandwidth) if bandwidth is not None else max(
        _silverman((X @ np.ones(p)) / np.sqrt(p)), 1e-3)
    y_signed = 2 * y - 1

    def loss(b):
        nrm = np.linalg.norm(b)
        if nrm < 1e-12:
            return 1e12
        bn = b / nrm
        z = X @ bn / h
        return -(y_signed * norm.cdf(z)).mean()

    def grad(b):
        nrm = np.linalg.norm(b)
        if nrm < 1e-12:
            return np.zeros_like(b)
        bn = b / nrm
        z = X @ bn / h
        phi = norm.pdf(z)
        # d/dbn = (1/n) sum y_signed * phi(z) * X_i / h
        g_bn = (y_signed[:, None] * phi[:, None] * X / h).mean(axis=0)
        # chain rule for b -> bn
        I = np.eye(p)
        J = (I - np.outer(bn, bn)) / nrm
        return -(J @ g_bn)

    beta0, *_ = np.linalg.lstsq(X, y_signed, rcond=None)
    nrm = np.linalg.norm(beta0)
    beta0 = beta0 / nrm if nrm > 1e-12 else np.ones(p) / np.sqrt(p)
    if beta0[0] < 0:
        beta0 = -beta0

    res = minimize(loss, beta0, jac=grad, method="BFGS",
                   options={"gtol": 1e-6, "maxiter": 200})
    beta_hat = res.x / max(np.linalg.norm(res.x), 1e-12)
    if beta_hat[0] < 0:
        beta_hat = -beta_hat

    # SE: outer-product-of-gradients approx (Horowitz 1992 Eq. 4)
    z = X @ beta_hat / h
    phi = norm.pdf(z)
    score_i = -(y_signed * phi)[:, None] * X / h
    info = score_i.T @ score_i / n
    try:
        cov = np.linalg.pinv(info) / n
    except np.linalg.LinAlgError:
        cov = np.full((p, p), np.nan)
    se = np.sqrt(np.maximum(np.diag(cov), 0))

    return RichResult(payload={
        "estimate": beta_hat.astype(float),
        "se": se.astype(float),
        "bandwidth": h, "n": n,
        "method": "Horowitz (1992) smoothed maximum-score",
    })


def cheatsheet():
    return "hrzb2: smoothed maximum-score estimator"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(7)
    n = 500
    X = rng.standard_normal((n, 2))
    beta = np.array([0.8, 0.6])
    y = (X @ beta + 0.5 * rng.standard_normal(n) > 0).astype(float)
    res = horowitz_smoothed_maximum_score(X, y)
    print(res)
    err = np.linalg.norm(res["estimate"] - beta)
    assert err < 0.4, f"err={err}"

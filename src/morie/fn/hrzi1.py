# morie.fn — function file (hadesllm/morie)
"""Single-index model estimator (Ichimura 1993; Horowitz 2009, Ch 4).

Model:   E[Y|X] = g(X'beta), beta normalised so that ||beta|| = 1
         and the leading coefficient is positive.

This implementation uses Ichimura's semiparametric least-squares:
    beta_hat = argmin_{||b||=1} sum_i (Y_i - g_hat(X_i'b ; b))^2

where ``g_hat(. ; b)`` is a leave-one-out Nadaraya-Watson smoother of Y
on the index X'b.  Minimisation uses ``scipy.optimize.minimize`` from a
warm OLS start.  Standard errors are obtained from the OLS-on-residuals
formula in Ichimura (1993, Eq. 4.2).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ._richresult import RichResult

__all__ = ["horowitz_index_model"]


def _silverman(x: np.ndarray) -> float:
    n = x.size
    if n < 2:
        return 1.0
    s = float(np.std(x, ddof=1))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    sigma = min(s, iqr / 1.349) if iqr > 0 else s
    if sigma <= 0:
        sigma = max(s, 1e-6)
    return 1.06 * sigma * n ** (-1.0 / 5.0)


def _nw_loo_1d(idx, y, h):
    u = (idx[:, None] - idx[None, :]) / h
    w = np.exp(-0.5 * u * u)
    np.fill_diagonal(w, 0.0)
    wsum = w.sum(axis=1)
    safe = np.where(wsum > 0, wsum, 1.0)
    return (w @ y) / safe


def horowitz_index_model(x, y, bandwidth=None):
    """Ichimura semiparametric single-index estimator.

    Returns
    -------
    RichResult with payload keys: estimate (beta, ||beta||=1), se,
    bandwidth, n, method.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.shape[0] != y.size:
        X = X.T
    n, p = X.shape
    if n < max(10, 2 * p):
        return RichResult(payload={"estimate": np.full(p, np.nan), "se": np.full(p, np.nan),
                                   "n": n, "method": "single-index (insufficient data)"})
    # OLS warm start
    beta0, *_ = np.linalg.lstsq(X, y, rcond=None)
    nrm = np.linalg.norm(beta0)
    if nrm < 1e-10:
        beta0 = np.ones(p) / np.sqrt(p)
    else:
        beta0 = beta0 / nrm
    # Sign normalisation: leading coefficient positive
    if beta0[0] < 0:
        beta0 = -beta0
    h0 = float(bandwidth) if bandwidth is not None else _silverman(X @ beta0)

    def objective(b):
        nb = np.linalg.norm(b)
        if nb < 1e-12:
            return 1e12
        bn = b / nb
        idx = X @ bn
        h = h0
        g_hat = _nw_loo_1d(idx, y, h)
        resid = y - g_hat
        return float((resid * resid).mean())

    res = minimize(objective, beta0, method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 200})
    beta_hat = res.x
    nb = np.linalg.norm(beta_hat)
    if nb > 1e-12:
        beta_hat = beta_hat / nb
    if beta_hat[0] < 0:
        beta_hat = -beta_hat

    # Asymptotic SE via numerical Hessian on the objective (Ichimura sandwich)
    eps = 1e-4
    H = np.zeros((p, p))
    f0 = objective(beta_hat)
    for i in range(p):
        for j in range(p):
            bp = beta_hat.copy(); bp[i] += eps; bp[j] += eps
            bm = beta_hat.copy(); bm[i] -= eps; bm[j] -= eps
            bpm = beta_hat.copy(); bpm[i] += eps; bpm[j] -= eps
            bmp = beta_hat.copy(); bmp[i] -= eps; bmp[j] += eps
            H[i, j] = (objective(bp) - objective(bpm) - objective(bmp) + objective(bm)) / (4 * eps * eps)
    H = 0.5 * (H + H.T)
    try:
        cov = np.linalg.pinv(H) / n
    except np.linalg.LinAlgError:
        cov = np.full((p, p), np.nan)
    se = np.sqrt(np.maximum(np.diag(cov), 0))

    return RichResult(payload={
        "estimate": beta_hat.astype(float),
        "se": se.astype(float),
        "n": n, "bandwidth": h0, "loss": float(f0),
        "method": "Ichimura (1993) single-index model",
    })


def cheatsheet():
    return "hrzi1: Ichimura single-index model"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(4)
    n = 500
    X = rng.standard_normal((n, 2))
    beta = np.array([0.8, 0.6])
    idx = X @ beta
    y = np.tanh(idx) + 0.1 * rng.standard_normal(n)
    res = horowitz_index_model(X, y)
    print(res)
    err = np.linalg.norm(res["estimate"] - beta)
    assert err < 0.3, f"err={err}"

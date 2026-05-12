# morie.fn — function file (hadesllm/morie)
"""Manski (1975) maximum-score estimator (Horowitz 2009, Ch 5).

    beta_hat = argmax_{||b||=1} (1/n) sum_i (2 Y_i - 1) * 1(X_i' b > 0)

Binary outcomes ``Y in {0,1}``; ``X`` is the regressor matrix.  Returns
a unit-norm coefficient vector.  Standard errors are NOT root-n
identifiable for Manski's estimator (Cube Root Asymptotics) so a
subsample SE is returned as a pragmatic stand-in.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ._richresult import RichResult

__all__ = ["horowitz_binary_response"]


def _score(beta, X, y_signed):
    return -(y_signed * (X @ beta > 0)).mean()


def horowitz_binary_response(x, y):
    """Manski maximum-score estimator (binary response).

    Returns
    -------
    RichResult with payload keys: estimate, se, n, method.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.shape[0] != y.size:
        X = X.T
    n, p = X.shape
    if n < max(10, 2 * p):
        return RichResult(payload={"estimate": np.full(p, np.nan),
                                   "se": np.full(p, np.nan), "n": n,
                                   "method": "maximum-score (insufficient data)"})
    y_signed = 2 * y - 1
    # Warm start from OLS / probit-like
    beta0, *_ = np.linalg.lstsq(X, y_signed, rcond=None)
    nrm = np.linalg.norm(beta0)
    beta0 = beta0 / nrm if nrm > 1e-12 else np.ones(p) / np.sqrt(p)
    if beta0[0] < 0:
        beta0 = -beta0

    # Multistart Nelder-Mead — the score function is piecewise constant
    rng = np.random.default_rng(0)
    best_beta = beta0; best_loss = _score(beta0, X, y_signed)
    starts = [beta0] + [rng.standard_normal(p) for _ in range(8)]
    for s in starts:
        s = s / max(np.linalg.norm(s), 1e-12)
        try:
            r = minimize(_score, s, args=(X, y_signed),
                         method="Nelder-Mead",
                         options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 300})
            b = r.x / max(np.linalg.norm(r.x), 1e-12)
            if b[0] < 0:
                b = -b
            l = _score(b, X, y_signed)
            if l < best_loss:
                best_loss = l; best_beta = b
        except Exception:
            continue

    # Subsample SE (50 reps, 50% subsample) since cube-root SE is not standard
    rng2 = np.random.default_rng(42)
    B = 30
    boot = np.zeros((B, p))
    m = max(20, n // 2)
    for b_idx in range(B):
        idx = rng2.choice(n, size=m, replace=False)
        Xb = X[idx]; yb_signed = y_signed[idx]
        s0 = best_beta + 0.05 * rng2.standard_normal(p)
        s0 = s0 / max(np.linalg.norm(s0), 1e-12)
        r = minimize(_score, s0, args=(Xb, yb_signed),
                     method="Nelder-Mead",
                     options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 150})
        bv = r.x / max(np.linalg.norm(r.x), 1e-12)
        if bv[0] < 0:
            bv = -bv
        # Cube-root: rescale subsample SE
        boot[b_idx] = bv
    se = boot.std(axis=0, ddof=1) * (m / n) ** (1.0 / 3.0)

    return RichResult(payload={
        "estimate": best_beta.astype(float),
        "se": se.astype(float),
        "score": float(-best_loss), "n": n,
        "method": "Manski (1975) maximum-score (binary response)",
        "warnings": ["Cube-root asymptotics: SEs from subsample rescaling, "
                     "not normal-theory."],
    })


def cheatsheet():
    return "hrzb1: Manski maximum-score estimator"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(6)
    n = 500
    X = rng.standard_normal((n, 2))
    beta = np.array([0.8, 0.6])
    y = (X @ beta + 0.5 * rng.standard_normal(n) > 0).astype(float)
    res = horowitz_binary_response(X, y)
    print(res)
    err = np.linalg.norm(res["estimate"] - beta)
    assert err < 0.4, f"err={err}"

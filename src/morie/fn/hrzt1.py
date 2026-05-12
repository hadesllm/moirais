# morie.fn — function file (hadesllm/morie)
"""Semiparametric ATE via kernel matching on propensity scores
(Heckman-Ichimura-Todd 1998; Horowitz 2009, Ch 9).

    ATE = E[Y(1) - Y(0)] estimated by

        (1/n1) sum_{i:D=1} Y_i
      - (1/n1) sum_{i:D=1} sum_{j:D=0} w_ij Y_j
        with w_ij ∝ K_h(p_i - p_j) / sum_k K_h(p_i - p_k).

The propensity score is fit by logistic regression.
"""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["horowitz_treatment_effect"]


def _logistic_newton(D, X, maxiter=50, tol=1e-8):
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(maxiter):
        eta = np.clip(X @ beta, -50, 50)
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = mu * (1 - mu)
        g = X.T @ (D - mu)
        H = X.T @ (X * W[:, None])
        try:
            step = np.linalg.solve(H + 1e-8 * np.eye(p), g)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H) @ g
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -50, 50)))


def _silverman(x):
    n = x.size
    if n < 2:
        return 1.0
    s = float(np.std(x, ddof=1))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    sigma = min(s, iqr / 1.349) if iqr > 0 else s
    if sigma <= 0:
        sigma = max(s, 1e-6)
    return 1.06 * sigma * n ** (-1.0 / 5.0)


def horowitz_treatment_effect(x, y, treatment, bandwidth=None, _bootstrap=True):
    """Heckman-Ichimura-Todd kernel-matching ATE.

    Parameters
    ----------
    x : array-like, shape (n, k)
        Covariates used to estimate the propensity score.
    y : array-like, shape (n,)
        Outcome.
    treatment : array-like, shape (n,)
        Binary treatment indicator (0/1).
    """
    y = np.asarray(y, dtype=float).ravel()
    D = np.asarray(treatment, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.shape[0] != y.size:
        X = X.T
    n = y.size
    if n < 30 or D.size != n or X.shape[0] != n:
        return RichResult(payload={"estimate": np.nan, "se": np.nan, "n": n,
                                   "method": "kernel-matching ATE (insufficient data)"})
    # Add an intercept if not present
    if not np.allclose(X[:, 0], 1.0):
        Xp = np.column_stack([np.ones(n), X])
    else:
        Xp = X
    e = _logistic_newton(D, Xp)
    e = np.clip(e, 1e-6, 1 - 1e-6)
    h = float(bandwidth) if bandwidth is not None else max(_silverman(e), 1e-3)
    t_idx = np.where(D > 0.5)[0]; c_idx = np.where(D < 0.5)[0]
    n_t = t_idx.size; n_c = c_idx.size
    if n_t < 2 or n_c < 2:
        return RichResult(payload={"estimate": np.nan, "se": np.nan, "n": n,
                                   "method": "kernel-matching ATE (one arm empty)"})
    # Kernel-matching counterfactual for each treated unit
    e_t = e[t_idx]; e_c = e[c_idx]
    u = (e_t[:, None] - e_c[None, :]) / h
    K = np.exp(-0.5 * u * u)
    w = K / np.maximum(K.sum(axis=1, keepdims=True), 1e-12)
    cf_treated = w @ y[c_idx]
    # Symmetrically counterfactual for controls
    u2 = (e_c[:, None] - e_t[None, :]) / h
    K2 = np.exp(-0.5 * u2 * u2)
    w2 = K2 / np.maximum(K2.sum(axis=1, keepdims=True), 1e-12)
    cf_control = w2 @ y[t_idx]
    # ATT and ATU; ATE is the population weighted average
    att = float((y[t_idx] - cf_treated).mean())
    atu = float((cf_control - y[c_idx]).mean())
    ate = (n_t * att + n_c * atu) / n
    # SE: bootstrap (50 reps) — disabled in nested calls to avoid recursion blow-up
    if _bootstrap:
        rng = np.random.default_rng(0)
        B = 50; boot = np.zeros(B)
        for b in range(B):
            idx = rng.integers(0, n, size=n)
            try:
                sub = horowitz_treatment_effect(X[idx], y[idx], D[idx],
                                                bandwidth=h, _bootstrap=False)
                boot[b] = sub["estimate"] if not np.isnan(sub["estimate"]) else ate
            except Exception:
                boot[b] = ate
        se = float(boot.std(ddof=1))
    else:
        se = float("nan")
    return RichResult(payload={
        "estimate": float(ate), "se": se,
        "att": att, "atu": atu, "bandwidth": h,
        "n": n, "n_treated": int(n_t), "n_control": int(n_c),
        "method": "Kernel-matching ATE (Heckman-Ichimura-Todd)",
    })


def cheatsheet():
    return "hrzt1: kernel-matching ATE"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(11)
    n = 600
    X = rng.standard_normal((n, 2))
    ps = 1.0 / (1.0 + np.exp(-X.sum(axis=1)))
    D = (rng.uniform(size=n) < ps).astype(float)
    y = 2 + 1.5 * D + X.sum(axis=1) + rng.standard_normal(n)
    res = horowitz_treatment_effect(X, y, D)
    print(res)
    assert abs(res["estimate"] - 1.5) < 0.5

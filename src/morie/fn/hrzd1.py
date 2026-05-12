# morie.fn — function file (hadesllm/morie)
"""Semiparametric duration/hazard model (Cox PH; Horowitz 2009, Ch 8).

    h(t | X) = h_0(t) * exp(X' beta)

Cox partial-likelihood estimator via Newton-Raphson; SE from the
observed information matrix.
"""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["horowitz_duration_model"]


def horowitz_duration_model(t, x, event):
    """Cox proportional-hazards estimator (partial likelihood)."""
    t = np.asarray(t, dtype=float).ravel()
    event = np.asarray(event, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.shape[0] != t.size:
        X = X.T
    n, p = X.shape
    if n < max(10, 2 * p) or t.size != n or event.size != n:
        return RichResult(payload={"estimate": np.full(p, np.nan),
                                   "se": np.full(p, np.nan), "n": n,
                                   "method": "Cox PH (insufficient data)"})
    # Sort by descending time (so cumulative sum runs over risk set)
    order = np.argsort(-t)
    Xs = X[order]; ev = event[order]
    beta = np.zeros(p)
    for _ in range(50):
        eta = Xs @ beta
        # clip for numerical safety
        eta = np.clip(eta, -50, 50)
        ehb = np.exp(eta)
        # risk-set sums S0[j] = sum_{i: t_i >= t_j} exp(eta_i)
        # Because we sorted descending, this is cumulative sum:
        S0 = np.cumsum(ehb)
        S1 = np.cumsum(Xs * ehb[:, None], axis=0)        # (n, p)
        # S2 only along diagonal direction; compute as cumulative outer
        # to avoid O(n^2). We compute the full p x p Hessian update:
        # Hessian = sum_j ev_j * (S2/S0 - (S1/S0)(S1/S0)')
        mean_X = S1 / np.maximum(S0[:, None], 1e-12)
        # Build cumulative S2 (n, p, p)
        outer = Xs[:, :, None] * Xs[:, None, :] * ehb[:, None, None]
        S2 = np.cumsum(outer, axis=0)
        # Score and Hessian only over events
        diff = Xs - mean_X
        score = (ev[:, None] * diff).sum(axis=0)
        var_X = S2 / np.maximum(S0[:, None, None], 1e-12) \
            - mean_X[:, :, None] * mean_X[:, None, :]
        # Observed information = -Hess(log-PL) = ∑ ev * var_X  (positive-definite)
        info = (ev[:, None, None] * var_X).sum(axis=0)
        try:
            step = np.linalg.solve(info + 1e-8 * np.eye(p), score)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(info) @ score
        # Damped Newton update for stability on weak risk sets
        damp = 1.0
        new_beta = beta + damp * step
        beta = new_beta
        if np.max(np.abs(step)) < 1e-6:
            break
    # cov(beta_hat) = Information^{-1}
    try:
        cov = np.linalg.pinv(info)
    except np.linalg.LinAlgError:
        cov = np.full((p, p), np.nan)
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    return RichResult(payload={
        "estimate": beta.astype(float) if beta.size > 1 else float(beta[0]),
        "se": se.astype(float) if se.size > 1 else float(se[0]),
        "n": n, "n_events": int(event.sum()),
        "method": "Cox proportional hazards (partial likelihood)",
    })


def cheatsheet():
    return "hrzd1: Cox PH semiparametric duration model"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(10)
    n = 600
    X = rng.standard_normal((n, 2))
    beta_true = np.array([0.5, -0.5])
    # Exponential durations with hazard ratio exp(X'beta)
    lam = np.exp(X @ beta_true)
    t = rng.exponential(1.0 / lam)
    ev = np.ones(n)
    res = horowitz_duration_model(t, X, ev)
    print(res)
    err = np.linalg.norm(res["estimate"] - beta_true)
    assert err < 0.3, f"err={err}"

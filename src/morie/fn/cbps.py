# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Covariate Balancing Propensity Score (CBPS)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def covariate_balance_ps(
    treatment: np.ndarray,
    covariates: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> DescriptiveResult:
    """CBPS: propensity scores that also balance covariates (Imai & Ratkovic, 2014).

    Simplified implementation using IRLS with balancing constraint.

    Parameters
    ----------
    treatment : (n,) binary
    covariates : (n, p) array

    Returns
    -------
    DescriptiveResult
    """
    t = np.asarray(treatment, dtype=float).ravel()
    X = np.asarray(covariates, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    X_int = np.column_stack([np.ones(n), X])
    k = X_int.shape[1]

    beta = np.zeros(k)
    for _ in range(max_iter):
        logit = X_int @ beta
        ps = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))

        w = ps * (1 - ps) + 1e-8
        grad_ll = X_int.T @ (t - ps)

        bal_weights = np.where(t == 1, 1 / (ps + 1e-8), -1 / (1 - ps + 1e-8))
        grad_bal = X_int.T @ (bal_weights * (t - ps))

        grad = grad_ll + grad_bal
        H = X_int.T @ (X_int * w[:, None])
        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        beta += 0.5 * delta
        if np.max(np.abs(delta)) < tol:
            break

    ps_final = 1 / (1 + np.exp(-np.clip(X_int @ beta, -30, 30)))

    smd_before = []
    smd_after = []
    ipw = np.where(t == 1, 1 / (ps_final + 1e-8), 1 / (1 - ps_final + 1e-8))
    for j in range(p):
        xj = X[:, j]
        smd_b = abs(xj[t == 1].mean() - xj[t == 0].mean()) / (xj.std() + 1e-8)
        wm1 = np.average(xj[t == 1], weights=ipw[t == 1])
        wm0 = np.average(xj[t == 0], weights=ipw[t == 0])
        smd_a = abs(wm1 - wm0) / (xj.std() + 1e-8)
        smd_before.append(float(smd_b))
        smd_after.append(float(smd_a))

    return DescriptiveResult(
        name="cbps",
        value=float(np.mean(smd_after)),
        extra={"smd_before": smd_before, "smd_after": smd_after, "n": n, "p": p, "ps_mean": float(ps_final.mean())},
    )


cbps = covariate_balance_ps


def cheatsheet() -> str:
    return "covariate_balance_ps({}) -> Covariate Balancing Propensity Score (CBPS)."

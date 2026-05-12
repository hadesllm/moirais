# morie.fn -- function file (hadesllm/morie)
"""Conditional logistic regression for matched data."""

from __future__ import annotations

import numpy as np
from scipy import optimize

from ._containers import RegressionResult


def conditional_logistic(
    y: np.ndarray,
    X: np.ndarray,
    strata: np.ndarray,
    *,
    max_iter: int = 200,
) -> RegressionResult:
    """Conditional logistic regression for matched case-control studies.

    The conditional likelihood eliminates stratum-specific intercepts,
    enabling consistent estimation with fixed-effects per stratum.

    Parameters
    ----------
    y : (n,) binary outcome {0, 1}
    X : (n, p) predictors (no intercept)
    strata : (n,) stratum/group identifiers
    max_iter : int

    Returns
    -------
    RegressionResult

    References
    ----------
    Breslow, N. E. & Day, N. E. (1980). *Statistical Methods in Cancer
    Research*, Vol. I. IARC.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    strata = np.asarray(strata).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape

    unique_strata = np.unique(strata)
    groups = []
    for s in unique_strata:
        idx = np.where(strata == s)[0]
        if np.sum(y[idx]) > 0 and np.sum(y[idx]) < len(idx):
            groups.append(idx)
    if len(groups) == 0:
        raise ValueError("No informative strata (need both cases and controls).")

    def neg_cond_loglik(beta):
        ll = 0.0
        for idx in groups:
            xb = X[idx] @ beta
            xb -= xb.max()
            exp_xb = np.exp(xb)
            cases = y[idx] == 1
            ll += np.sum(xb[cases]) - np.sum(cases) * np.log(np.sum(exp_xb) + 1e-300)
        return -ll

    def grad(beta):
        g = np.zeros(p)
        for idx in groups:
            xb = X[idx] @ beta
            xb -= xb.max()
            exp_xb = np.exp(xb)
            prob = exp_xb / (np.sum(exp_xb) + 1e-300)
            cases = y[idx] == 1
            n_cases = np.sum(cases)
            g += np.sum(X[idx][cases], axis=0) - n_cases * (X[idx].T @ prob)
        return g

    x0 = np.zeros(p)
    res = optimize.minimize(neg_cond_loglik, x0, jac=lambda b: -grad(b),
                            method="BFGS", options={"maxiter": max_iter})
    beta = res.x
    ll = float(-res.fun)

    try:
        if hasattr(res, "hess_inv") and isinstance(res.hess_inv, np.ndarray):
            se_arr = np.sqrt(np.diag(res.hess_inv).clip(0))
        else:
            se_arr = np.full(p, float("nan"))
    except Exception:
        se_arr = np.full(p, float("nan"))

    from scipy import stats as _st
    z_vals = beta / (se_arr + 1e-300)
    p_vals = 2.0 * _st.norm.sf(np.abs(z_vals))

    names = [f"x{j}" for j in range(p)]
    return RegressionResult(
        method="Conditional Logistic",
        coefficients={nm: float(b) for nm, b in zip(names, beta)},
        se={nm: float(s) for nm, s in zip(names, se_arr)},
        p_values={nm: float(pv) for nm, pv in zip(names, p_vals)},
        n=n,
        k=p,
        extra={
            "n_strata": len(groups),
            "log_likelihood": ll,
        },
    )


clogr = conditional_logistic


def cheatsheet() -> str:
    return "conditional_logistic({}) -> Conditional logistic for matched data."

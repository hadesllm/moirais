# moirais.fn — function file (hadesllm/moirais)
"""The only true wisdom is in knowing you know nothing. — Socrates"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats

from ._containers import RegressionResult


def cox_ph(time, event, x) -> RegressionResult:
    """Cox PH model via partial likelihood maximization.

    Parameters
    ----------
    time : array-like
        Observed survival times.
    event : array-like
        Event indicator (1 = event, 0 = censored).
    x : array-like
        Covariate matrix (n x p) or vector (n,).

    Returns
    -------
    RegressionResult
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] == 1 and X.shape[1] > 1:
        X = X.T
    n, p = X.shape
    order = np.argsort(-time)  # descending for risk sets
    time_s, event_s, X_s = time[order], event[order], X[order]

    def neg_partial_ll(beta):
        risk = np.exp(np.clip(X_s @ beta, -500, 500))
        cum_risk = np.cumsum(risk)
        ll = np.sum(event_s * (X_s @ beta - np.log(np.maximum(cum_risk, 1e-300))))
        return -ll

    beta0 = np.zeros(p)
    res = optimize.minimize(neg_partial_ll, beta0, method="L-BFGS-B")
    beta = res.x
    # Hessian for SE
    risk = np.exp(np.clip(X_s @ beta, -500, 500))
    cum_risk = np.cumsum(risk)
    H = np.zeros((p, p))
    for i in range(n):
        if event_s[i]:
            if cum_risk[i] > 0:
                weighted_x = np.cumsum(X_s[: i + 1] * risk[: i + 1, None], axis=0)[i]
                mean_x = weighted_x / cum_risk[i]
                for j1 in range(p):
                    for j2 in range(p):
                        wx2 = np.sum(X_s[: i + 1, j1] * X_s[: i + 1, j2] * risk[: i + 1]) / cum_risk[i]
                        H[j1, j2] += wx2 - mean_x[j1] * mean_x[j2]
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.full((p, p), np.nan)
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    names = [f"x{i}" for i in range(p)]
    coefficients = {nm: float(b) for nm, b in zip(names, beta)}
    se_dict = {nm: float(s) for nm, s in zip(names, se)}
    p_dict = {
        nm: float(2 * stats.norm.sf(abs(beta[i] / se[i]))) if se[i] > 0 else float("nan") for i, nm in enumerate(names)
    }
    hr = {nm: float(np.exp(beta[i])) for i, nm in enumerate(names)}
    return RegressionResult(
        method="Cox PH",
        coefficients=coefficients,
        se=se_dict,
        p_values=p_dict,
        n=n,
        k=p,
        extra={
            "hazard_ratios": hr,
            "log_partial_likelihood": float(-res.fun),
            "converged": res.success,
        },
    )


cox = cox_ph


def cheatsheet() -> str:
    return "The only true wisdom is in knowing you know nothing. — Socrates"

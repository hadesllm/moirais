# morie.fn — function file (hadesllm/morie)
"""Cox regression semiparametric (partial likelihood)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

__all__ = ["cxreg"]


def cxreg(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray,
    *,
    alpha: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> dict[str, Any]:
    r"""
    Fit a Cox proportional hazards model via partial likelihood.

    The partial likelihood is:

    .. math::

        L(\beta) = \prod_{i: \delta_i=1}
        \frac{\exp(\beta^T X_i)}{\sum_{j \in R(t_i)} \exp(\beta^T X_j)}

    where :math:`R(t_i)` is the risk set at time :math:`t_i`.

    :param time: Event/censoring times, shape (n,).
    :param event: Event indicators (1=event, 0=censored), shape (n,).
    :param X: Covariate matrix, shape (n, p).
    :param alpha: Significance level. Default 0.05.
    :param max_iter: Max Newton-Raphson iterations. Default 100.
    :param tol: Convergence tolerance. Default 1e-8.
    :return: Dict with ``beta``, ``se``, ``hr`` (hazard ratios), ``ci_lower``,
        ``ci_upper``, ``log_partial_likelihood``, ``n``, ``n_events``, ``converged``.
    :raises ValueError: If arrays are mismatched or empty.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 11. Springer.
    Cox, D.R. (1972). JRSS B, 34(2), 187-220.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    if n == 0:
        raise ValueError("Arrays must be non-empty.")
    if len(time) != n or len(event) != n:
        raise ValueError("time, event, X must have same number of rows.")

    order = np.argsort(-time)
    time = time[order]
    event = event[order]
    X = X[order]

    beta = np.zeros(p)
    converged = False

    for it in range(max_iter):
        eta = X @ beta
        exp_eta = np.exp(eta - np.max(eta))

        cum_exp = np.cumsum(exp_eta)
        cum_X_exp = np.cumsum(X * exp_eta[:, None], axis=0)

        score = np.zeros(p)
        hessian = np.zeros((p, p))

        for i in range(n):
            if event[i] == 0:
                continue
            denom = cum_exp[i]
            if denom < 1e-300:
                continue
            x_bar = cum_X_exp[i] / denom
            score += X[i] - x_bar
            for j in range(p):
                for k in range(p):
                    hessian[j, k] -= (np.sum(X[:i+1, j] * X[:i+1, k] * exp_eta[:i+1]) / denom - x_bar[j] * x_bar[k])

        try:
            step = np.linalg.solve(hessian - 1e-8 * np.eye(p), -score)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, -score, rcond=None)[0]

        beta += step
        if np.max(np.abs(step)) < tol:
            converged = True
            break

    try:
        info = -hessian
        var = np.linalg.inv(info + 1e-8 * np.eye(p))
        se = np.sqrt(np.maximum(np.diag(var), 0))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)

    eta_final = X @ beta
    exp_eta_final = np.exp(eta_final - np.max(eta_final))
    cum_final = np.cumsum(exp_eta_final)
    log_pl = float(np.sum(event * (eta_final - np.log(np.maximum(cum_final, 1e-300)))))

    z = stats.norm.ppf(1.0 - alpha / 2.0)
    return {
        "beta": beta,
        "se": se,
        "hr": np.exp(beta),
        "ci_lower": beta - z * se,
        "ci_upper": beta + z * se,
        "log_partial_likelihood": log_pl,
        "n": n,
        "n_events": int(np.sum(event)),
        "converged": converged,
    }


def cheatsheet() -> str:
    return "cxreg({time, event, X}) -> Cox regression via partial likelihood."

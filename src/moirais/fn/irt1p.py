# moirais.fn — function file (hadesllm/moirais)
"""1-Parameter Logistic (Rasch) IRT model via joint MLE."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moirais.fn._containers import IRTResult


def irt1p(
    data: pd.DataFrame | np.ndarray,
    *,
    theta_grid: np.ndarray | None = None,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> IRTResult:
    """Fit a 1-Parameter Logistic (Rasch) IRT model.

    Estimates item difficulty (b) parameters via joint MLE with fixed
    discrimination a=1 for all items.  Ability (theta) is estimated
    simultaneously via Newton-Raphson on the conditional likelihood.

    Parameters
    ----------
    data : DataFrame or ndarray
        Binary item response matrix (n_respondents x n_items).
        Values should be 0/1.
    theta_grid : ndarray, optional
        Grid for information computation.  Default linspace(-4, 4, 81).
    max_iter : int
        Maximum EM iterations (default 500).
    tol : float
        Convergence tolerance for parameter change (default 1e-6).

    Returns
    -------
    IRTResult
        model="1PL", item_params={item: {"a": 1.0, "b": float}},
        theta (person ability estimates), se_theta, fit dict.

    References
    ----------
    Rasch, G. (1960). Probabilistic Models for Some Intelligence and
    Attainment Tests. Danish Institute for Educational Research.
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape

    if k < 2:
        raise ValueError("Need at least 2 items for IRT model.")

    # Replace NaN with 0 for computation, track missingness
    missing = np.isnan(X)
    X = np.where(missing, 0.0, X)

    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"item_{j}" for j in range(k)]

    # Initial difficulty estimates from item proportions
    p_item = np.clip(X.mean(axis=0), 0.001, 0.999)
    b = -np.log(p_item / (1.0 - p_item))  # logit of proportion correct

    # Initial theta from total score
    total = X.sum(axis=1)
    p_person = np.clip(total / k, 0.001, 0.999)
    theta = np.log(p_person / (1.0 - p_person))

    # Joint MLE via alternating Newton-Raphson
    loglik_prev = -np.inf
    for iteration in range(max_iter):
        # E-step: compute probabilities P(X=1 | theta, b) with a=1
        # P = 1 / (1 + exp(-(theta_i - b_j)))
        logits = theta[:, None] - b[None, :]
        P = 1.0 / (1.0 + np.exp(-np.clip(logits, -700, 700)))
        P = np.clip(P, 1e-10, 1.0 - 1e-10)

        # Log-likelihood
        loglik = np.sum(X * np.log(P) + (1.0 - X) * np.log(1.0 - P))

        if abs(loglik - loglik_prev) < tol:
            break
        loglik_prev = loglik

        # Update b: for fixed theta, MLE of b_j
        # Score: d(logL)/d(b_j) = sum_i (P_ij - X_ij)
        # Hessian: d2(logL)/d(b_j)^2 = -sum_i P_ij * (1 - P_ij)
        for j in range(k):
            score_b = np.sum(P[:, j] - X[:, j])
            info_b = np.sum(P[:, j] * (1.0 - P[:, j]))
            if info_b > 1e-10:
                b[j] += score_b / info_b

        # Update theta: for fixed b, MLE of theta_i
        for i in range(n):
            logits_i = theta[i] - b
            Pi = 1.0 / (1.0 + np.exp(-np.clip(logits_i, -700, 700)))
            Pi = np.clip(Pi, 1e-10, 1.0 - 1e-10)
            score_t = np.sum(X[i, :] - Pi)
            info_t = np.sum(Pi * (1.0 - Pi))
            if info_t > 1e-10:
                theta[i] += score_t / info_t

        # Center theta for identifiability
        theta -= theta.mean()

    # Standard errors of theta via observed information
    logits_final = theta[:, None] - b[None, :]
    P_final = 1.0 / (1.0 + np.exp(-np.clip(logits_final, -700, 700)))
    P_final = np.clip(P_final, 1e-10, 1.0 - 1e-10)
    info_theta = np.sum(P_final * (1.0 - P_final), axis=1)
    se_theta = np.where(info_theta > 1e-10, 1.0 / np.sqrt(info_theta), np.nan)

    # Fit statistics
    n_params = k + n  # b params + theta params
    aic = -2.0 * loglik + 2.0 * n_params
    bic = -2.0 * loglik + np.log(n * k) * n_params

    item_params = {}
    for j, name in enumerate(names):
        item_params[name] = {"a": 1.0, "b": float(b[j])}

    # Test information on grid
    if theta_grid is None:
        theta_grid = np.linspace(-4, 4, 81)
    info_grid = np.zeros_like(theta_grid)
    for j in range(k):
        logits_g = theta_grid - b[j]
        Pg = 1.0 / (1.0 + np.exp(-np.clip(logits_g, -700, 700)))
        info_grid += Pg * (1.0 - Pg)  # a=1 so I = P*Q

    return IRTResult(
        model="1PL",
        item_params=item_params,
        theta=theta,
        se_theta=se_theta,
        fit={"loglik": float(loglik), "aic": float(aic), "bic": float(bic), "n_iter": iteration + 1, "n": n, "k": k},
        info=info_grid,
    )


rasch = irt1p


def cheatsheet() -> str:
    return "irt1p({}) -> 1-Parameter Logistic (Rasch) IRT model via joint MLE."

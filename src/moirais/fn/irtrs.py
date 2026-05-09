# moirais.fn — function file (hadesllm/moirais)
"""Rating Scale Model for Likert-type items."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moirais.fn._containers import IRTResult


def irt_rsm(
    data: pd.DataFrame | np.ndarray,
    *,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> IRTResult:
    """Fit Andrich's Rating Scale Model (RSM).

    All items share common threshold parameters; only item location
    (difficulty) varies across items.

    Parameters
    ----------
    data : DataFrame or ndarray
        Polytomous response matrix (n x k), integer values 0..m.
    max_iter : int
        Maximum iterations (default 300).
    tol : float
        Convergence tolerance (default 1e-6).

    Returns
    -------
    IRTResult
        model="RSM", item_params with delta_j (location) and shared tau.

    References
    ----------
    Andrich, D. (1978). A rating formulation for ordered response categories.
    Psychometrika, 43(4), 561-573.
    """
    X = np.asarray(data, dtype=np.float64)
    X = np.where(np.isnan(X), 0, X)
    n, k = X.shape
    if k < 2:
        raise ValueError("Need at least 2 items.")

    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"item_{j}" for j in range(k)]
    m = int(X.max())
    if m < 1:
        raise ValueError("Max response must be >= 1.")

    item_means = X.mean(axis=0)
    delta = -(item_means - item_means.mean())
    tau = np.linspace(-1, 1, m)

    total = X.sum(axis=1)
    theta = (total - total.mean()) / max(total.std(), 1e-10)

    for _ in range(max_iter):
        old_delta = delta.copy()
        for j in range(k):
            residual = X[:, j].mean() - _rsm_expected(theta, delta[j], tau)
            delta[j] -= 0.3 * residual
        if np.max(np.abs(delta - old_delta)) < tol:
            break

    item_params = {}
    for j, name in enumerate(names):
        item_params[name] = {"delta": float(delta[j]), "tau": tau.tolist()}

    se = np.full(n, 1.0 / np.sqrt(max(k * m * 0.25, 1)))

    return IRTResult(
        model="RSM",
        item_params=item_params,
        theta=theta,
        se_theta=se,
        fit={"n": n, "k": k, "m": m},
    )


def _rsm_expected(theta, delta, tau):
    """Expected score under RSM for given theta, delta, tau."""
    m = len(tau)
    cum_tau = np.cumsum(tau)
    numer = 0.0
    denom = 0.0
    for cat in range(m + 1):
        if cat == 0:
            log_p = 0.0
        else:
            log_p = cat * (theta.mean() - delta) - cum_tau[cat - 1]
        exp_p = np.exp(np.clip(log_p, -500, 500))
        numer += cat * exp_p
        denom += exp_p
    return numer / max(denom, 1e-10)


rsm = irt_rsm


def cheatsheet() -> str:
    return "irt_rsm({}) -> Rating Scale Model for Likert-type items."

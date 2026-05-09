# moirais.fn — function file (hadesllm/moirais)
"""IRT calibration pipeline (JMLE for 1PL/2PL)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _icc_2pl(theta: float, a: float, b: float) -> float:
    """2PL ICC."""
    z = a * (theta - b)
    if z > 500:
        return 1.0
    if z < -500:
        return 0.0
    return 1.0 / (1.0 + np.exp(-z))


def irt_calibrate(
    data: pd.DataFrame | np.ndarray,
    *,
    model: str = "2PL",
    max_iter: int = 100,
    tol: float = 1e-4,
) -> dict:
    """Joint Maximum Likelihood Estimation for IRT calibration.

    Estimates item parameters and person abilities jointly. Supports
    1PL (Rasch) and 2PL models on binary (0/1) data.

    Parameters
    ----------
    data : DataFrame or ndarray
        Binary item responses (n x k). Rows = persons, columns = items.
    model : str
        '1PL' or '2PL' (default '2PL').
    max_iter : int
        Maximum EM iterations (default 100).
    tol : float
        Convergence tolerance on parameter change (default 1e-4).

    Returns
    -------
    dict
        Keys: 'item_params' ({item: {a, b}}), 'theta' (ndarray),
        'se_theta' (ndarray), 'converged', 'n_iter', 'model'.

    References
    ----------
    Baker, F. B. & Kim, S. H. (2004). Item Response Theory: Parameter
    Estimation Techniques (2nd ed.). Marcel Dekker.
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape
    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"item_{j + 1}" for j in range(k)]

    # Initialise
    p_obs = np.nanmean(X, axis=0)
    p_obs = np.clip(p_obs, 0.01, 0.99)
    b = -np.log(p_obs / (1 - p_obs))  # logit of difficulty
    a = np.ones(k) if model == "1PL" else np.full(k, 1.0)

    # Theta from sum scores
    raw_scores = np.nansum(X, axis=1)
    raw_scores = np.clip(raw_scores, 0.5, k - 0.5)
    theta = np.log(raw_scores / (k - raw_scores))

    converged = False
    n_iter = 0

    for it in range(max_iter):
        n_iter = it + 1
        old_b = b.copy()
        old_a = a.copy()

        # Update b (and a for 2PL) via Newton-Raphson per item
        for j in range(k):
            p_vec = np.array([_icc_2pl(theta[i], a[j], b[j]) for i in range(n)])
            p_vec = np.clip(p_vec, 1e-10, 1 - 1e-10)

            # Observed vs expected
            valid = ~np.isnan(X[:, j])
            resid = X[valid, j] - p_vec[valid]
            w = p_vec[valid] * (1 - p_vec[valid])

            # Update b
            grad_b = -a[j] * np.sum(resid)
            hess_b = a[j] ** 2 * np.sum(w)
            if abs(hess_b) > 1e-10:
                b[j] -= grad_b / hess_b

            # Update a (2PL only)
            if model == "2PL":
                grad_a = np.sum(resid * (theta[valid] - b[j]))
                hess_a = np.sum(w * (theta[valid] - b[j]) ** 2)
                if abs(hess_a) > 1e-10:
                    a[j] += grad_a / hess_a
                    a[j] = max(a[j], 0.1)  # floor

        # Update theta via Newton-Raphson per person
        for i in range(n):
            p_vec = np.array([_icc_2pl(theta[i], a[j], b[j]) for j in range(k)])
            p_vec = np.clip(p_vec, 1e-10, 1 - 1e-10)
            valid = ~np.isnan(X[i, :])
            resid = X[i, valid] - p_vec[valid]
            w = p_vec[valid] * (1 - p_vec[valid])
            a_valid = a[valid]

            grad = np.sum(a_valid * resid)
            hess = np.sum(a_valid**2 * w)
            if abs(hess) > 1e-10:
                theta[i] += grad / hess

        # Check convergence
        max_change = max(np.max(np.abs(b - old_b)), np.max(np.abs(a - old_a)))
        if max_change < tol:
            converged = True
            break

    # SE of theta
    se_theta = np.zeros(n)
    for i in range(n):
        info = 0.0
        for j in range(k):
            if np.isnan(X[i, j]):
                continue
            p = _icc_2pl(theta[i], a[j], b[j])
            p = np.clip(p, 1e-10, 1 - 1e-10)
            info += a[j] ** 2 * p * (1 - p)
        se_theta[i] = 1.0 / np.sqrt(info) if info > 1e-10 else np.nan

    item_params = {}
    for j in range(k):
        item_params[names[j]] = {"a": float(a[j]), "b": float(b[j])}

    return {
        "item_params": item_params,
        "theta": theta,
        "se_theta": se_theta,
        "converged": converged,
        "n_iter": n_iter,
        "model": model,
    }


def cheatsheet() -> str:
    return "_icc_2pl({}) -> IRT calibration pipeline (JMLE for 1PL/2PL)."

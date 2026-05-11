# morie.fn — function file (hadesllm/morie)
"""MLE theta estimation."""

from __future__ import annotations

import numpy as np
from ._richresult import RichResult


def _icc_2pl(theta: float, a: float, b: float) -> float:
    """2PL item characteristic curve."""
    exp_val = a * (theta - b)
    if exp_val > 500:
        return 1.0
    if exp_val < -500:
        return 0.0
    return 1.0 / (1.0 + np.exp(-exp_val))


def irt_mle_theta(
    responses: np.ndarray,
    item_params: dict,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> dict:
    """Maximum Likelihood Estimation of theta (ability) via Newton-Raphson.

    Parameters
    ----------
    responses : ndarray
        Binary response vector (k,) for one person.
    item_params : dict
        {item_name: {'a': ..., 'b': ...}} ordered same as responses.
    max_iter : int
        Maximum Newton-Raphson iterations (default 50).
    tol : float
        Convergence tolerance (default 1e-6).

    Returns
    -------
    dict
        Keys: 'theta', 'se', 'converged', 'n_iter'.

    References
    ----------
    Lord, F. M. (1980). Applications of Item Response Theory to Practical
    Testing Problems. Lawrence Erlbaum.
    """
    r = np.asarray(responses, dtype=np.float64).ravel()
    params_list = list(item_params.values())
    k = len(params_list)
    if len(r) != k:
        raise ValueError(f"responses length {len(r)} != n_items {k}")

    theta = 0.0
    converged = False
    n_iter = 0

    for it in range(max_iter):
        n_iter = it + 1
        numerator = 0.0
        denominator = 0.0
        for j in range(k):
            if np.isnan(r[j]):
                continue
            a = params_list[j].get("a", 1.0)
            b = params_list[j].get("b", 0.0)
            p = _icc_2pl(theta, a, b)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            numerator += a * (r[j] - p)
            denominator += a**2 * p * (1 - p)

        if abs(denominator) < 1e-15:
            break
        delta = numerator / denominator
        theta += delta
        if abs(delta) < tol:
            converged = True
            break

    # SE from Fisher information
    info = 0.0
    for j in range(k):
        if np.isnan(r[j]):
            continue
        a = params_list[j].get("a", 1.0)
        b = params_list[j].get("b", 0.0)
        p = _icc_2pl(theta, a, b)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        info += a**2 * p * (1 - p)

    se = 1.0 / np.sqrt(info) if info > 1e-15 else np.nan

    return RichResult(payload={"theta": float(theta), "se": float(se), "converged": converged, "n_iter": n_iter})


def cheatsheet() -> str:
    return "_icc_2pl({}) -> MLE theta estimation."

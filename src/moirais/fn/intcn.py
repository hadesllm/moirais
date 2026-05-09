# moirais.fn — function file (hadesllm/moirais)
"""Interval censoring likelihood (Turnbull estimator)."""

from __future__ import annotations

import numpy as np

__all__ = ["intcn"]


def intcn(
    left: np.ndarray,
    right: np.ndarray,
    *,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> dict:
    """Turnbull NPMLE for interval-censored data.

    Parameters
    ----------
    left : array-like
        Left endpoints of intervals (n,). Use 0 for left-censored.
    right : array-like
        Right endpoints of intervals (n,). Use np.inf for right-censored.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    dict
        times, survival, mass, n_obs, n_iter, converged.
    """
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    n = len(left)

    finite_r = right[np.isfinite(right)]
    all_times = np.unique(np.concatenate([left, finite_r]))
    all_times = np.sort(all_times[all_times >= 0])
    m = len(all_times)

    if m == 0:
        return {"times": np.array([]), "survival": np.array([1.0]),
                "mass": np.array([]), "n_obs": n, "n_iter": 0, "converged": True}

    p = np.ones(m) / m
    converged = False
    n_iter = 0

    for it in range(max_iter):
        alpha = np.zeros((n, m))
        for i in range(n):
            in_interval = (all_times >= left[i]) & (all_times <= right[i])
            denom = np.sum(p[in_interval])
            if denom > 0:
                alpha[i, in_interval] = p[in_interval] / denom

        p_new = np.sum(alpha, axis=0) / n
        p_new = np.maximum(p_new, 0)
        total = np.sum(p_new)
        if total > 0:
            p_new /= total

        n_iter = it + 1
        if np.max(np.abs(p_new - p)) < tol:
            converged = True
            p = p_new
            break
        p = p_new

    surv = 1 - np.cumsum(p)

    return {
        "times": all_times,
        "survival": surv,
        "mass": p,
        "n_obs": n,
        "n_iter": n_iter,
        "converged": converged,
    }


intcn_fn = intcn


def cheatsheet() -> str:
    return "intcn(left, right) -> Turnbull NPMLE for interval-censored data."

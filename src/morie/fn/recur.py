# morie.fn — function file (hadesllm/morie)
"""Recurrent event analysis (Andersen-Gill model)."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["recur"]


def recur(start: np.ndarray, stop: np.ndarray, event: np.ndarray, X: np.ndarray, cdf=None, *, max_iter: int = 100, tol: float = 1e-8) -> dict:
    """Andersen-Gill model for recurrent events.

    Extension of Cox model for counting process data with
    multiple events per subject.

    Parameters
    ----------
    start : array-like
        Left endpoint of interval (n,).
    stop : array-like
        Right endpoint of interval (n,).
    event : array-like
        Event indicator at stop (1=event, 0=censored) (n,).
    X : array-like
        Covariate matrix (n, p).
    max_iter : int
        Maximum Newton-Raphson iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    dict
        coefficients, se, hazard_ratios, p_values, converged,
        n_obs, n_events.
    """
    start = np.asarray(start, dtype=float)
    stop = np.asarray(stop, dtype=float)
    event = np.asarray(event, dtype=float)
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    event_times = np.unique(stop[event == 1])
    beta = np.zeros(p)
    converged = False

    for _ in range(max_iter):
        score = np.zeros(p)
        hess = np.zeros((p, p))

        for tj in event_times:
            at_risk = (start < tj) & (stop >= tj)
            died = (stop == tj) & (event == 1)
            if not np.any(at_risk):
                continue

            X_r = X[at_risk]
            eta_r = X_r @ beta
            eta_r -= eta_r.max()
            exp_r = np.exp(eta_r)
            s0 = np.sum(exp_r)
            s1 = exp_r @ X_r
            s2 = (X_r * exp_r[:, None]).T @ X_r
            dj = np.sum(died)

            score += np.sum(X[died], axis=0) - dj * s1 / s0
            hess -= dj * (s2 / s0 - np.outer(s1 / s0, s1 / s0))

        try:
            step = np.linalg.solve(hess, score)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, score, rcond=None)[0]

        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < tol:
            converged = True
            beta = beta_new
            break
        beta = beta_new

    try:
        se = np.sqrt(np.diag(np.linalg.inv(-hess)))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)

    z = beta / np.where(se > 0, se, np.nan)
    pvals = 2 * (1 - norm.cdf(np.abs(z)))

    return {
        "coefficients": beta,
        "se": se,
        "hazard_ratios": np.exp(beta),
        "p_values": pvals,
        "converged": converged,
        "n_obs": n,
        "n_events": int(np.sum(event)),
    }


recur_fn = recur


def cheatsheet() -> str:
    return "recur(start, stop, event, X) -> Andersen-Gill recurrent event model."

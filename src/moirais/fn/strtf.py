"""Stratified Cox proportional hazards model."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["strtf"]


def strtf(time: np.ndarray, event: np.ndarray, X: np.ndarray, strata: np.ndarray, cdf=None, *, max_iter: int = 100, tol: float = 1e-8) -> dict:
    """Stratified Cox model with stratum-specific baseline hazards.

    Parameters
    ----------
    time : array-like
        Observed event/censoring times (n,).
    event : array-like
        Event indicator (1=event, 0=censored) (n,).
    X : array-like
        Covariate matrix (n, p).
    strata : array-like
        Stratum indicator (n,).
    max_iter : int
        Maximum Newton-Raphson iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    dict
        coefficients, se, hazard_ratios, p_values, converged,
        n_strata, n_obs, n_events.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    X = np.asarray(X, dtype=float)
    strata = np.asarray(strata)
    n, p = X.shape
    unique_strata = np.unique(strata)

    beta = np.zeros(p)
    converged = False

    for _ in range(max_iter):
        score = np.zeros(p)
        hess = np.zeros((p, p))

        for s_val in unique_strata:
            mask = strata == s_val
            t_s = time[mask]
            e_s = event[mask]
            X_s = X[mask]
            ns = len(t_s)
            order = np.argsort(-t_s)
            t_o = t_s[order]
            e_o = e_s[order]
            X_o = X_s[order]

            eta = X_o @ beta
            eta -= eta.max()
            exp_eta = np.exp(eta)
            cum_exp = np.cumsum(exp_eta)
            cum_Xexp = np.cumsum(X_o * exp_eta[:, None], axis=0)

            for i in range(ns):
                if e_o[i] == 0:
                    continue
                w = cum_Xexp[i] / cum_exp[i]
                score += X_o[i] - w
                hess -= np.outer(X_o[i] * exp_eta[i], X_o[i]) / cum_exp[i]
                hess += np.outer(w, w)

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
        "n_strata": len(unique_strata),
        "n_obs": n,
        "n_events": int(np.sum(event)),
    }


strtf_fn = strtf


def cheatsheet() -> str:
    return "strtf(time, event, X, strata) -> Stratified Cox model."

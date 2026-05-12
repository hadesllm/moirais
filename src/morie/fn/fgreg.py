# morie.fn -- function file (hadesllm/morie)
"""Fine-Gray competing risks regression."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["fgreg"]


def fgreg(time: np.ndarray, event: np.ndarray, X: np.ndarray, cdf=None, *, cause: int = 1, max_iter: int = 100, tol: float = 1e-8) -> dict:
    """Fine-Gray subdistribution hazard regression.

    Models the subdistribution hazard for a specific cause
    in competing risks data.

    Parameters
    ----------
    time : array-like
        Observed event/censoring times (n,).
    event : array-like
        Event type (0=censored, cause=event of interest,
        other=competing event) (n,).
    X : array-like
        Covariate matrix (n, p).
    cause : int
        Cause of interest.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    dict
        coefficients, se, subdist_hazard_ratios, p_values,
        converged, n_obs, n_events.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    is_cause = (event == cause).astype(float)
    is_competing = ((event > 0) & (event != cause)).astype(float)
    is_censored = (event == 0).astype(float)

    order = np.argsort(-time)
    t_s = time[order]
    cause_s = is_cause[order]
    comp_s = is_competing[order]
    X_s = X[order]

    from .km import kaplan_meier
    km_cens = kaplan_meier(time, (1 - is_competing).astype(int))
    G_func = np.interp(time, km_cens.times, km_cens.survival,
                       left=1.0, right=km_cens.survival[-1])

    w = np.ones(n)
    for i in range(n):
        if is_competing[i] == 1:
            w[i] = G_func[i] / max(G_func[i], 1e-10)

    beta = np.zeros(p)
    converged = False

    for _ in range(max_iter):
        eta = X @ beta
        eta -= eta.max()
        exp_eta = np.exp(eta)
        score = np.zeros(p)
        hess = np.zeros((p, p))

        event_times = np.unique(time[is_cause == 1])
        for tj in event_times:
            at_risk = (time >= tj) | (is_competing == 1)
            died = (time == tj) & (is_cause == 1)
            if not np.any(at_risk):
                continue

            w_r = w[at_risk] * exp_eta[at_risk]
            s0 = np.sum(w_r)
            if s0 == 0:
                continue
            s1 = (w_r @ X[at_risk]) / s0
            dj = np.sum(died)
            X_d = X[died]
            score += np.sum(X_d, axis=0) - dj * s1

            s2 = (X[at_risk] * w_r[:, None]).T @ X[at_risk] / s0
            hess -= dj * (s2 - np.outer(s1, s1))

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
        "subdist_hazard_ratios": np.exp(beta),
        "p_values": pvals,
        "converged": converged,
        "n_obs": n,
        "n_events": int(np.sum(is_cause)),
    }


fgreg_fn = fgreg


def cheatsheet() -> str:
    return "fgreg(time, event, X) -> Fine-Gray competing risks regression."

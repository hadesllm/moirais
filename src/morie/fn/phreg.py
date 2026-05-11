# morie.fn — function file (hadesllm/morie)
"""Piecewise constant hazard regression."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

__all__ = ["phreg"]


def phreg(time: np.ndarray, event: np.ndarray, X: np.ndarray, cdf=None, *, breaks: np.ndarray | None = None, n_intervals: int = 5) -> dict:
    """Piecewise constant hazard regression model.

    Parameters
    ----------
    time : array-like
        Observed event/censoring times (n,).
    event : array-like
        Event indicator (1=event, 0=censored) (n,).
    X : array-like
        Covariate matrix (n, p).
    breaks : array-like, optional
        Interval boundaries. Default: quantiles of event times.
    n_intervals : int
        Number of intervals if breaks not given.

    Returns
    -------
    dict
        coefficients, baseline_hazards, breaks, se, p_values, n_obs, n_events.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    if breaks is None:
        event_t = time[event == 1]
        if len(event_t) < n_intervals:
            breaks = np.array([0, np.max(time)])
        else:
            q = np.linspace(0, 100, n_intervals + 1)
            breaks = np.unique(np.percentile(event_t, q))
    else:
        breaks = np.asarray(breaks, dtype=float)

    K = len(breaks) - 1

    def neg_loglik(params):
        beta = params[:p]
        log_lam = params[p:]
        lp = X @ beta
        ll = 0.0
        for k in range(K):
            lo, hi = breaks[k], breaks[k + 1]
            in_interval = (time > lo) & (time <= hi)
            at_risk = time > lo
            d_k = np.sum(in_interval & (event == 1))
            exposure = np.minimum(time[at_risk], hi) - lo
            lam_k = np.exp(log_lam[k])
            ll += d_k * log_lam[k] + d_k * np.mean(lp[in_interval & (event == 1)]) if d_k > 0 else 0
            ll -= lam_k * np.sum(np.exp(lp[at_risk]) * exposure)
        return -ll

    x0 = np.zeros(p + K)
    result = minimize(neg_loglik, x0, method="L-BFGS-B")

    beta = result.x[:p]
    log_lam = result.x[p:]
    baseline_haz = np.exp(log_lam)

    try:
        from scipy.optimize import approx_fprime
        hess_diag = np.zeros(p + K)
        eps = 1e-5
        for i in range(p + K):
            def f_i(xi):
                params = result.x.copy()
                params[i] = xi
                return neg_loglik(params)
            h = approx_fprime([result.x[i]], f_i, eps)[0]
            hess_diag[i] = max(h, 1e-10)
        se = 1.0 / np.sqrt(hess_diag[:p])
    except Exception:
        se = np.full(p, np.nan)

    z = beta / np.where(se > 0, se, np.nan)
    pvals = 2 * (1 - norm.cdf(np.abs(z)))

    return {
        "coefficients": beta,
        "baseline_hazards": baseline_haz,
        "breaks": breaks,
        "se": se,
        "p_values": pvals,
        "n_obs": n,
        "n_events": int(np.sum(event)),
    }


phreg_fn = phreg


def cheatsheet() -> str:
    return "phreg(time, event, X) -> Piecewise constant hazard regression."

# moirais.fn — function file (hadesllm/moirais)
"""Joinpoint survival regression."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

__all__ = ["jnpnt"]


def jnpnt(
    time: np.ndarray,
    event: np.ndarray,
    *,
    max_joinpoints: int = 3,
) -> dict:
    """Joinpoint regression for survival hazard trends.

    Fits piecewise log-linear hazard models and selects the
    number of joinpoints by BIC.

    Parameters
    ----------
    time : array-like
        Observed event/censoring times (n,).
    event : array-like
        Event indicator (1=event, 0=censored) (n,).
    max_joinpoints : int
        Maximum number of joinpoints to test.

    Returns
    -------
    dict
        n_joinpoints, joinpoints, slopes, intercepts,
        bic, log_likelihood, n_obs, n_events.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    n = len(time)
    event_times = np.sort(time[event == 1])
    d = len(event_times)

    if d < 4:
        return {
            "n_joinpoints": 0, "joinpoints": np.array([]),
            "slopes": np.array([0.0]), "intercepts": np.array([np.log(d / np.sum(time) + 1e-300)]),
            "bic": np.inf, "log_likelihood": 0.0, "n_obs": n, "n_events": int(d),
        }

    def fit_model(jp_list):
        breaks = np.concatenate([[0], np.sort(jp_list), [np.max(time)]])
        k = len(breaks) - 1
        params = np.zeros(2 * k)

        def neg_ll(p):
            ll = 0.0
            for seg in range(k):
                lo, hi = breaks[seg], breaks[seg + 1]
                mask_event = (event_times > lo) & (event_times <= hi)
                mask_risk = (time > lo)
                a, b = p[2 * seg], p[2 * seg + 1]
                t_ev = event_times[mask_event]
                if len(t_ev) > 0:
                    ll += np.sum(a + b * t_ev)
                t_r = np.minimum(time[mask_risk], hi)
                exp_hi = np.clip(a + b * t_r, -500, 500)
                exp_lo_val = np.clip(a + b * lo, -500, 500)
                lam_int = np.sum(np.exp(exp_hi) - np.exp(exp_lo_val))
                ll -= lam_int / (abs(b) + 1e-10)
            return -ll if np.isfinite(ll) else 1e20

        result = minimize(neg_ll, params, method="Nelder-Mead",
                          options={"maxiter": 3000})
        npar = 2 * k
        ll = -result.fun
        bic = -2 * ll + npar * np.log(n)
        slopes = result.x[1::2]
        intercepts = result.x[0::2]
        return ll, bic, slopes, intercepts

    best_bic = np.inf
    best = None

    for njp in range(max_joinpoints + 1):
        if njp == 0:
            ll, bic, slopes, intercepts = fit_model([])
            if bic < best_bic:
                best_bic = bic
                best = (0, np.array([]), slopes, intercepts, bic, ll)
        else:
            candidates = np.percentile(event_times, np.linspace(10, 90, min(10, d)))
            from itertools import combinations
            for jp in combinations(candidates, njp):
                ll, bic, slopes, intercepts = fit_model(list(jp))
                if bic < best_bic:
                    best_bic = bic
                    best = (njp, np.array(jp), slopes, intercepts, bic, ll)

    return {
        "n_joinpoints": best[0],
        "joinpoints": best[1],
        "slopes": best[2],
        "intercepts": best[3],
        "bic": float(best[4]),
        "log_likelihood": float(best[5]),
        "n_obs": n,
        "n_events": int(np.sum(event)),
    }


jnpnt_fn = jnpnt


def cheatsheet() -> str:
    return "jnpnt(time, event) -> Joinpoint survival regression."

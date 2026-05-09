# moirais.fn — function file (hadesllm/moirais)
"""Restricted mean survival time difference."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["rmstd"]


def rmstd(time: np.ndarray, event: np.ndarray, group: np.ndarray, cdf=None, *, tau: float | None = None, alpha: float = 0.05) -> dict:
    """Restricted mean survival time (RMST) difference between two groups.

    RMST is the area under the Kaplan-Meier curve up to tau.

    Parameters
    ----------
    time : array-like
        Observed event/censoring times.
    event : array-like
        Event indicator (1=event, 0=censored).
    group : array-like
        Group indicator (two distinct values).
    tau : float, optional
        Restriction time. Default: minimum of max observed times.
    alpha : float
        Significance level.

    Returns
    -------
    dict
        rmst_0, rmst_1, difference, se, ci_lower, ci_upper, p_value.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    group = np.asarray(group)
    groups = np.unique(group)
    if len(groups) != 2:
        raise ValueError("Exactly two groups required.")

    def _rmst(t, e, tau_val):
        order = np.argsort(t, kind="stable")
        t_s, e_s = t[order], e[order]
        unique_t = np.unique(t_s[e_s == 1])
        unique_t = unique_t[unique_t <= tau_val]
        all_t = np.concatenate([[0], unique_t, [tau_val]])
        s = 1.0
        surv_vals = [1.0]
        gw = 0.0
        gw_vals = [0.0]

        for tj in unique_t:
            nj = np.sum(t_s >= tj)
            dj = np.sum((t_s == tj) & (e_s == 1))
            if nj > 0:
                s *= 1 - dj / nj
                if nj > dj:
                    gw += dj / (nj * (nj - dj))
            surv_vals.append(s)
            gw_vals.append(gw)
        surv_vals.append(s)
        gw_vals.append(gw)

        area = 0.0
        var_area = 0.0
        for i in range(len(all_t) - 1):
            dt = all_t[i + 1] - all_t[i]
            area += surv_vals[i] * dt
            var_area += (surv_vals[i] ** 2) * gw_vals[i] * (dt ** 2)

        return area, var_area

    g0 = group == groups[0]
    g1 = group == groups[1]
    if tau is None:
        tau = min(np.max(time[g0]), np.max(time[g1]))

    rmst0, var0 = _rmst(time[g0], event[g0], tau)
    rmst1, var1 = _rmst(time[g1], event[g1], tau)

    diff = rmst1 - rmst0
    se = np.sqrt(var0 + var1)
    z = norm.ppf(1 - alpha / 2)

    return {
        "rmst_0": float(rmst0),
        "rmst_1": float(rmst1),
        "difference": float(diff),
        "se": float(se),
        "ci_lower": float(diff - z * se),
        "ci_upper": float(diff + z * se),
        "p_value": float(2 * (1 - norm.cdf(abs(diff / se)))) if se > 0 else 1.0,
        "tau": float(tau),
    }


rmstd_fn = rmstd


def cheatsheet() -> str:
    return "rmstd(time, event, group) -> Restricted mean survival time difference."

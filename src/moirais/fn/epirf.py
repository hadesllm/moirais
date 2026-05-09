# moirais.fn — function file (hadesllm/moirais)
"""Effective reproduction number Rt (Wallinga-Teunis method)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as _st


def effective_rt(
    incidence: np.ndarray,
    serial_interval_pmf: np.ndarray,
    *,
    tau: int = 7,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Estimate time-varying reproduction number Rt (Wallinga-Teunis).

    For each case at time *t*, the relative likelihood that it was
    infected by a case at time *s* is proportional to the serial
    interval distribution evaluated at *t - s*. Rt for time *s* is the
    sum of these likelihoods over all subsequent cases.

    .. math::

        p_{ij} = \\frac{w(t_i - t_j)}{\\sum_{k \\neq i} w(t_i - t_k)}

        R_t = \\sum_{i} p_{ij}

    Parameters
    ----------
    incidence : array_like
        Daily incidence counts (integer or float).
    serial_interval_pmf : array_like
        Probability mass function of the serial interval (starting at
        lag 1). Must sum to approximately 1.
    tau : int, default 7
        Smoothing window for the posterior mean (not used if <= 1).
    alpha : float, default 0.05
        Significance level for credible intervals.

    Returns
    -------
    dict
        Keys: 'Rt' (array), 'ci_lower', 'ci_upper', 't'.

    References
    ----------
    Wallinga, J. & Teunis, P. (2004). Different epidemic curves for
    severe acute respiratory syndrome reveal similar impacts of control
    measures. American Journal of Epidemiology, 160(6), 509-516.
    """
    inc = np.asarray(incidence, dtype=float)
    w = np.asarray(serial_interval_pmf, dtype=float)

    if inc.ndim != 1 or inc.size < 2:
        raise ValueError("incidence must be a 1-D array with >= 2 elements.")
    if w.ndim != 1 or w.size < 1:
        raise ValueError("serial_interval_pmf must be a non-empty 1-D array.")

    T = len(inc)
    max_lag = len(w)

    rt = np.full(T, np.nan)

    for s in range(T):
        num = 0.0
        for t in range(s + 1, min(s + max_lag + 1, T)):
            lag = t - s - 1
            if lag < len(w):
                denom = 0.0
                for k in range(max(0, t - max_lag), t):
                    lag_k = t - k - 1
                    if lag_k < len(w):
                        denom += inc[k] * w[lag_k]
                if denom > 0:
                    num += inc[t] * w[lag] / denom
        rt[s] = num

    if tau > 1:
        kernel = np.ones(tau) / tau
        rt_smooth = np.convolve(rt, kernel, mode="same")
    else:
        rt_smooth = rt.copy()

    z = _st.norm.ppf(1 - alpha / 2)
    valid = ~np.isnan(rt_smooth)
    se = np.full(T, np.nan)
    se[valid] = np.sqrt(np.maximum(rt_smooth[valid], 0))
    ci_lo = np.where(valid, rt_smooth - z * se, np.nan)
    ci_hi = np.where(valid, rt_smooth + z * se, np.nan)
    ci_lo = np.maximum(ci_lo, 0)

    return {
        "Rt": rt_smooth,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "t": np.arange(T),
    }


epirf = effective_rt


def cheatsheet() -> str:
    return "effective_rt({}) -> Effective reproduction number Rt (Wallinga-Teunis)."

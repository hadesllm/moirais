# morie.fn — function file (hadesllm/morie)
"""Effective reproduction number Rt (EpiEstim-style)."""

from __future__ import annotations

import numpy as np
import scipy.stats as stats


def rt_effective(
    incidence: list[int] | np.ndarray,
    serial_interval: list[float] | np.ndarray | None = None,
    window: int = 7,
    prior_mean: float = 5.0,
    prior_sd: float = 5.0,
    confidence: float = 0.95,
) -> dict:
    """Estimate the effective reproduction number Rt over time.

    Uses the Bayesian method of Cori et al. (2013) (EpiEstim). A Gamma
    prior on Rt is updated with the likelihood of observed incidence
    given a discretized serial interval distribution.

    Parameters
    ----------
    incidence : array-like of int
        Daily new case counts.
    serial_interval : array-like of float, optional
        Probability mass for serial interval (day 0, 1, 2, ...).
        Default: discretized Gamma(mean=4.7, sd=2.9).
    window : int, default 7
        Sliding window width (days).
    prior_mean : float, default 5.0
        Prior mean for Rt (Gamma prior).
    prior_sd : float, default 5.0
        Prior SD for Rt.
    confidence : float, default 0.95
        Credible interval level.

    Returns
    -------
    dict
        Keys: 't' (time indices), 'Rt_mean', 'Rt_lower', 'Rt_upper',
        'Rt_median'.

    References
    ----------
    Cori, A. et al. (2013). A new framework and software to estimate
    time-varying reproduction numbers during epidemics.
    American Journal of Epidemiology, 178(9), 1505-1512.
    """
    inc = np.asarray(incidence, dtype=float)
    n = len(inc)

    if serial_interval is None:
        si_mean, si_sd = 4.7, 2.9
        shape = (si_mean / si_sd) ** 2
        scale = si_sd**2 / si_mean
        max_si = min(20, n)
        si = np.array([stats.gamma.pdf(k, a=shape, scale=scale) for k in range(max_si)])
        si[0] = 0.0
        si = si / si.sum() if si.sum() > 0 else si
    else:
        si = np.asarray(serial_interval, dtype=float)

    prior_shape = (prior_mean / prior_sd) ** 2
    prior_rate = prior_mean / prior_sd**2

    t_out = []
    rt_mean = []
    rt_lo = []
    rt_hi = []
    rt_med = []
    alpha_q = (1 - confidence) / 2

    for t_end in range(window, n):
        t_start = t_end - window

        lambda_t = np.zeros(window)
        for s in range(window):
            day = t_start + s + 1
            conv = 0.0
            for j in range(1, min(len(si), day + 1)):
                conv += si[j] * inc[day - j] if day - j >= 0 else 0.0
            lambda_t[s] = conv

        sum_inc = np.sum(inc[t_start + 1 : t_end + 1])
        sum_lambda = np.sum(lambda_t)

        post_shape = prior_shape + sum_inc
        post_rate = prior_rate + sum_lambda

        if post_rate > 0:
            mean_val = post_shape / post_rate
            lo = stats.gamma.ppf(alpha_q, a=post_shape, scale=1.0 / post_rate)
            hi = stats.gamma.ppf(1 - alpha_q, a=post_shape, scale=1.0 / post_rate)
            med = stats.gamma.ppf(0.5, a=post_shape, scale=1.0 / post_rate)
        else:
            mean_val = lo = hi = med = np.nan

        t_out.append(t_end)
        rt_mean.append(float(mean_val))
        rt_lo.append(float(lo))
        rt_hi.append(float(hi))
        rt_med.append(float(med))

    return {
        "t": np.array(t_out),
        "Rt_mean": np.array(rt_mean),
        "Rt_lower": np.array(rt_lo),
        "Rt_upper": np.array(rt_hi),
        "Rt_median": np.array(rt_med),
    }


rtefv = rt_effective


def cheatsheet() -> str:
    return "rt_effective({}) -> Effective reproduction number Rt (EpiEstim-style)."

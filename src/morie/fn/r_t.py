# morie.fn -- function file (hadesllm/morie)
"""Real-time effective reproduction number (Rt) estimation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def realtime_rt(
    incidence: np.ndarray | list,
    *,
    serial_mean: float = 5.0,
    serial_sd: float = 2.0,
    window: int = 7,
    prior_a: float = 1.0,
    prior_b: float = 5.0,
) -> DescriptiveResult:
    """
    Estimate the real-time effective reproduction number Rt using a
    Bayesian framework similar to Bettencourt & Ribeiro (2008).

    Uses a Gamma prior and Poisson likelihood over a sliding window.

    Parameters
    ----------
    incidence : array-like
        Daily case counts.
    serial_mean : float
        Mean serial interval (days).
    serial_sd : float
        SD of serial interval.
    window : int
        Sliding window size for smoothing.
    prior_a, prior_b : float
        Gamma prior shape and rate.

    Returns
    -------
    DescriptiveResult
        With extra keys 'rt', 'ci_lower', 'ci_upper'.

    References
    ----------
    Bettencourt, L. M., & Ribeiro, R. M. (2008). Real time Bayesian
    estimation of the epidemic potential of emerging infectious diseases.
    *PLoS ONE*, 3(5), e2185.
    """
    inc = np.asarray(incidence, dtype=float)
    if inc.ndim != 1:
        raise ValueError("incidence must be 1-D.")
    n = len(inc)
    if n < window + 1:
        raise ValueError("incidence must have length > window + 1.")

    k = serial_mean**2 / serial_sd**2
    theta = serial_sd**2 / serial_mean
    max_si = int(serial_mean + 3 * serial_sd)
    from scipy.stats import gamma as gamma_dist

    si_pmf = np.zeros(max_si + 1)
    for j in range(1, max_si + 1):
        si_pmf[j] = gamma_dist.pdf(j, a=k, scale=theta)
    si_pmf = si_pmf / si_pmf.sum() if si_pmf.sum() > 0 else si_pmf

    lam = np.zeros(n)
    for t in range(1, n):
        for j in range(1, min(t, max_si) + 1):
            lam[t] += inc[t - j] * si_pmf[j]

    rt_vals = np.full(n, np.nan)
    ci_lo = np.full(n, np.nan)
    ci_hi = np.full(n, np.nan)

    for t in range(window, n):
        sum_inc = inc[t - window + 1 : t + 1].sum()
        sum_lam = lam[t - window + 1 : t + 1].sum()
        if sum_lam > 0:
            post_a = prior_a + sum_inc
            post_b = prior_b + sum_lam
            rt_vals[t] = post_a / post_b
            ci_lo[t] = gamma_dist.ppf(0.025, a=post_a, scale=1.0 / post_b)
            ci_hi[t] = gamma_dist.ppf(0.975, a=post_a, scale=1.0 / post_b)

    median_rt = float(np.nanmedian(rt_vals[window:]))

    return DescriptiveResult(
        name="Rt",
        value=median_rt,
        extra={"rt": rt_vals, "ci_lower": ci_lo, "ci_upper": ci_hi},
    )


r_t = realtime_rt


def cheatsheet() -> str:
    return "realtime_rt({}) -> Real-time effective reproduction number (Rt) estimation."

# morie.fn -- function file (hadesllm/morie)
"""Effective reproduction number Rt."""

import numpy as np

from ._containers import DescriptiveResult


def effective_rt(
    incidence: np.ndarray,
    serial_interval: float = 5.0,
    window: int = 7,
) -> DescriptiveResult:
    r"""Estimate the effective reproduction number Rt.

    Uses the ratio method with serial interval correction:

    .. math::

        R_t = \\frac{I_t}{\\sum_{s=1}^{w} I_{t-s} \\cdot g_s}

    where :math:`g_s` is a discretized serial interval distribution
    (here approximated as uniform over the window, scaled by
    serial_interval).

    For a simple approximation, Rt ~ I(t) / I(t-1) when serial interval
    equals the reporting interval.

    Parameters
    ----------
    incidence : array-like
        Daily (or per-period) incidence counts.
    serial_interval : float, default 5.0
        Mean serial interval in same time units as incidence.
    window : int, default 7
        Smoothing window for denominator.

    Returns
    -------
    DescriptiveResult
        value = array of Rt values (length = len(incidence) - window).

    References
    ----------
    Cori, A., Ferguson, N. M., Fraser, C., & Cauchemez, S. (2013). A new
    framework and software to estimate time-varying reproduction numbers
    during epidemics. American Journal of Epidemiology, 178(9), 1505-1512.
    """
    inc = np.asarray(incidence, dtype=float)
    n = len(inc)
    if n <= window:
        raise ValueError("incidence must be longer than window")

    rt_vals = np.full(n, np.nan)
    for t in range(window, n):
        denom = np.sum(inc[t - window : t])
        if denom > 0:
            rt_vals[t] = inc[t] * (serial_interval / window) / (denom / window)
        else:
            rt_vals[t] = np.nan

    rt_out = rt_vals[window:]

    return DescriptiveResult(
        name="Effective Rt",
        value=rt_out,
        extra={
            "serial_interval": serial_interval,
            "window": window,
            "mean_rt": float(np.nanmean(rt_out)) if np.any(np.isfinite(rt_out)) else None,
        },
    )


r_eff = effective_rt


def cheatsheet() -> str:
    return "effective_rt({}) -> Effective reproduction number Rt."

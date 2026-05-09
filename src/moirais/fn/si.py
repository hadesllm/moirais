"""Serial interval estimation from case pairs."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import ESRes


def serial_interval(
    onset_primary: np.ndarray | list,
    onset_secondary: np.ndarray | list,
) -> ESRes:
    """
    Estimate the serial interval distribution from paired onset dates.

    The serial interval is the time between symptom onset in a primary
    case and symptom onset in the secondary case they infected.

    Parameters
    ----------
    onset_primary : array-like
        Onset times/days for primary cases.
    onset_secondary : array-like
        Onset times/days for corresponding secondary cases.

    Returns
    -------
    ESRes
        estimate = mean serial interval; extra has 'sd', 'median',
        'gamma_shape', 'gamma_scale'.

    References
    ----------
    Vink, M. A., Bootsma, M. C., & Wallinga, J. (2014). Serial
    intervals of respiratory infectious diseases: a systematic review
    and analysis. *Am J Epidemiol*, 180(9), 865-875.
    """
    p = np.asarray(onset_primary, dtype=float)
    s = np.asarray(onset_secondary, dtype=float)
    if len(p) != len(s):
        raise ValueError("Primary and secondary onset arrays must be same length.")
    if len(p) < 2:
        raise ValueError("Need at least 2 pairs.")

    intervals = s - p
    mean_si = float(np.mean(intervals))
    sd_si = float(np.std(intervals, ddof=1))
    median_si = float(np.median(intervals))

    positive = intervals[intervals > 0]
    shape, loc, scale = (np.nan, np.nan, np.nan)
    if len(positive) >= 3:
        shape, loc, scale = stats.gamma.fit(positive, floc=0)

    se = sd_si / np.sqrt(len(intervals))
    ci_lo = mean_si - 1.96 * se
    ci_hi = mean_si + 1.96 * se

    return ESRes(
        measure="serial_interval",
        estimate=mean_si,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        se=se,
        n=len(intervals),
        extra={
            "sd": sd_si,
            "median": median_si,
            "gamma_shape": float(shape),
            "gamma_scale": float(scale),
        },
    )


si = serial_interval


def cheatsheet() -> str:
    return "serial_interval({}) -> Serial interval estimation from case pairs."

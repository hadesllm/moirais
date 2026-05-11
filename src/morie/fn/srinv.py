"""Serial interval estimation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as _st


def serial_interval(
    intervals: np.ndarray,
    distribution: str = "gamma",
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Estimate the serial interval distribution.

    The serial interval is the time between symptom onset in a primary
    case and symptom onset in a secondary case. Unlike generation time,
    serial intervals can be negative (pre-symptomatic transmission).

    Parameters
    ----------
    intervals : array_like
        Observed serial intervals (may include negative values for
        pre-symptomatic transmission).
    distribution : str, default "gamma"
        "gamma" (shifted to allow negatives), "normal", or "lognormal"
        (positive-only).
    alpha : float, default 0.05
        Significance level for CI on the mean.

    Returns
    -------
    dict
        Keys: 'mean', 'sd', 'ci_lower', 'ci_upper', 'params',
              'distribution', 'n', 'prop_negative'.

    References
    ----------
    Nishiura, H., Linton, N. M., & Akhmetzhanov, A. R. (2020). Serial
    interval of novel coronavirus (COVID-19) infections. International
    Journal of Infectious Diseases, 93, 284-286.
    """
    intervals = np.asarray(intervals, dtype=float)
    if intervals.size == 0:
        raise ValueError("intervals must not be empty.")

    n = len(intervals)
    prop_neg = float(np.mean(intervals < 0))

    if distribution == "normal":
        mu = float(np.mean(intervals))
        sigma = float(np.std(intervals, ddof=1))
        params = {"mu": mu, "sigma": sigma}
    elif distribution == "gamma":
        shift = 0.0
        if np.any(intervals <= 0):
            shift = float(np.min(intervals)) - 0.01
        shifted = intervals - shift
        a, _, scale = _st.gamma.fit(shifted, floc=0)
        mu = a * scale + shift
        sigma = float(np.sqrt(a * scale**2))
        params = {"shape": float(a), "scale": float(scale), "shift": float(shift)}
    elif distribution == "lognormal":
        pos = intervals[intervals > 0]
        if len(pos) < 2:
            raise ValueError("Need >= 2 positive intervals for lognormal fit.")
        s, _, scale = _st.lognorm.fit(pos, floc=0)
        mu_ln = np.log(scale)
        mu = float(np.exp(mu_ln + s**2 / 2))
        sigma = float(np.sqrt((np.exp(s**2) - 1) * np.exp(2 * mu_ln + s**2)))
        params = {"mu_log": float(mu_ln), "sigma_log": float(s)}
    else:
        raise ValueError("distribution must be 'gamma', 'normal', or 'lognormal'.")

    se = sigma / np.sqrt(n) if n > 0 else 0.0
    z = _st.norm.ppf(1 - alpha / 2)

    return {
        "mean": mu,
        "sd": sigma,
        "ci_lower": float(mu - z * se),
        "ci_upper": float(mu + z * se),
        "params": params,
        "distribution": distribution,
        "n": n,
        "prop_negative": prop_neg,
    }


srinv = serial_interval


def cheatsheet() -> str:
    return "serial_interval({}) -> Serial interval distribution estimation."

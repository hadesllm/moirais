# morie.fn — function file (hadesllm/morie)
"""Nowcasting with reporting delay adjustment."""

from __future__ import annotations

from typing import Any

import numpy as np


def nowcast(reported: np.ndarray, delay_pmf: np.ndarray, cdf=None, *, max_delay: int | None = None) -> dict[str, Any]:
    """Adjust epidemic incidence for reporting delays (nowcasting).

    Uses the reporting delay distribution to inflate recent counts
    that are likely incomplete.

    .. math::

        N_t^{\\text{adj}} = \\frac{N_t^{\\text{reported}}}{F(T - t)}

    where F is the CDF of the reporting delay distribution evaluated
    at the time remaining until the present.

    Parameters
    ----------
    reported : array_like
        Reported daily incidence counts (most recent at end).
    delay_pmf : array_like
        PMF of reporting delay (delay_pmf[k] = P(delay = k days),
        starting at k=0).
    max_delay : int or None
        Maximum reporting delay to consider. If None, uses
        len(delay_pmf) - 1.

    Returns
    -------
    dict
        Keys: 'nowcast' (adjusted counts), 'reporting_completeness',
              'inflation_factor'.

    References
    ----------
    Hohle, M. & an der Heiden, M. (2014). Bayesian nowcasting during
    the STEC O104:H4 outbreak in Germany, 2011. Biometrics, 70(4),
    993-1002.
    """
    rep = np.asarray(reported, dtype=float)
    pmf = np.asarray(delay_pmf, dtype=float)

    if rep.ndim != 1:
        raise ValueError("reported must be 1-D.")
    if pmf.ndim != 1 or pmf.size < 1:
        raise ValueError("delay_pmf must be non-empty 1-D array.")
    if np.any(pmf < 0):
        raise ValueError("delay_pmf values must be non-negative.")

    pmf = pmf / pmf.sum()

    if max_delay is None:
        max_delay = len(pmf) - 1

    cdf = np.cumsum(pmf)
    T = len(rep)

    nowcast_arr = rep.copy()
    completeness = np.ones(T)
    inflation = np.ones(T)

    max_inflation = 10.0
    for i in range(T):
        days_available = T - 1 - i
        if days_available < max_delay:
            idx = min(days_available, len(cdf) - 1)
            comp = cdf[idx]
            completeness[i] = comp
            if comp > 0.01:
                inflation[i] = min(1.0 / comp, max_inflation)
                nowcast_arr[i] = rep[i] * inflation[i]
            else:
                inflation[i] = max_inflation
                nowcast_arr[i] = rep[i] * max_inflation

    return {
        "nowcast": nowcast_arr,
        "reporting_completeness": completeness,
        "inflation_factor": inflation,
    }


nowcs = nowcast


def cheatsheet() -> str:
    return "nowcast({}) -> Nowcasting with reporting delay adjustment."

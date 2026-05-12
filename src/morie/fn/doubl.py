# morie.fn -- function file (hadesllm/morie)
"""Epidemic doubling time estimation."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def doubling_time(
    r: float | None = None,
    *,
    incidence: np.ndarray | list | None = None,
    alpha: float = 0.05,
) -> ESRes:
    r"""
    Compute the epidemic doubling time from the exponential growth rate.

    .. math::

        T_d = \\frac{\\ln 2}{r}

    If *r* is not provided, it is estimated from the incidence series
    via log-linear regression.

    Parameters
    ----------
    r : float, optional
        Exponential growth rate (per unit time).
    incidence : array-like, optional
        Case counts; used to estimate r if not given.
    alpha : float
        Significance level for CI.

    Returns
    -------
    ESRes

    References
    ----------
    Vynnycky, E., & White, R. G. (2010). *An Introduction to
    Infectious Disease Modelling*. Oxford University Press, Ch. 2.
    """
    from scipy import stats as _st

    if r is None and incidence is None:
        raise ValueError("Provide either r or incidence.")

    se_r = None
    n = None
    if r is None:
        inc = np.asarray(incidence, dtype=float)
        pos = inc > 0
        if pos.sum() < 3:
            raise ValueError("Need at least 3 positive incidence values.")
        t_vals = np.arange(len(inc))[pos]
        log_inc = np.log(inc[pos])
        slope, intercept, r_val, p_val, se_slope = _st.linregress(t_vals, log_inc)
        r = slope
        se_r = se_slope
        n = int(pos.sum())

    if r <= 0:
        raise ValueError("Growth rate r must be positive for doubling time.")

    td = np.log(2) / r
    ci_lo, ci_hi = None, None
    se_td = None
    if se_r is not None and se_r > 0:
        z = _st.norm.ppf(1 - alpha / 2)
        se_td = np.log(2) * se_r / (r**2)
        ci_lo = td - z * se_td
        ci_hi = td + z * se_td

    return ESRes(
        measure="doubling_time",
        estimate=float(td),
        ci_lower=float(ci_lo) if ci_lo is not None else None,
        ci_upper=float(ci_hi) if ci_hi is not None else None,
        se=float(se_td) if se_td is not None else None,
        n=n,
        extra={"growth_rate": float(r)},
    )


doubl = doubling_time


def cheatsheet() -> str:
    return "doubling_time({}) -> Epidemic doubling time estimation."

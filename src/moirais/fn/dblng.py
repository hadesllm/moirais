# moirais.fn — function file (hadesllm/moirais)
"""Doubling time from epidemic growth rate."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as _st


def doubling_time(
    growth_rate: float | None = None,
    incidence: np.ndarray | None = None,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compute epidemic doubling time from growth rate or incidence.

    .. math::

        T_d = \\frac{\\ln 2}{r}

    Parameters
    ----------
    growth_rate : float or None
        Pre-computed exponential growth rate. If None, estimated from
        *incidence*.
    incidence : array_like or None
        Daily incidence counts (used if growth_rate is None).
    alpha : float, default 0.05
        Significance level for CI on doubling time.

    Returns
    -------
    dict
        Keys: 'doubling_time', 'ci_lower', 'ci_upper', 'growth_rate'.

    References
    ----------
    Vynnycky, E. & White, R. (2010). *An Introduction to Infectious
    Disease Modelling*. Oxford University Press, Ch. 2.
    """
    if growth_rate is None and incidence is None:
        raise ValueError("Provide either growth_rate or incidence.")

    if growth_rate is None:
        inc = np.asarray(incidence, dtype=float)
        mask = inc > 0
        t_vals = np.where(mask)[0].astype(float)
        y_vals = inc[mask]
        if len(t_vals) < 3:
            raise ValueError("Need >= 3 positive-incidence points.")
        slope, _, _, _, se = _st.linregress(t_vals, np.log(y_vals))
        r = slope
        r_se = se
    else:
        r = float(growth_rate)
        r_se = None

    if r <= 0:
        return {
            "doubling_time": np.inf,
            "ci_lower": np.inf,
            "ci_upper": np.inf,
            "growth_rate": float(r),
        }

    td = np.log(2) / r

    if r_se is not None and r_se > 0:
        z = _st.norm.ppf(1 - alpha / 2)
        r_lo = r - z * r_se
        r_hi = r + z * r_se
        td_lo = np.log(2) / r_hi if r_hi > 0 else np.inf
        td_hi = np.log(2) / r_lo if r_lo > 0 else np.inf
    else:
        td_lo = np.nan
        td_hi = np.nan

    return {
        "doubling_time": float(td),
        "ci_lower": float(td_lo),
        "ci_upper": float(td_hi),
        "growth_rate": float(r),
    }


dblng = doubling_time


def cheatsheet() -> str:
    return "doubling_time({}) -> Epidemic doubling time from growth rate."

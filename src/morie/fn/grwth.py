# morie.fn — function file (hadesllm/morie)
"""Exponential growth rate estimation from epidemic incidence."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as _st


def exponential_growth_rate(
    incidence: np.ndarray,
    *,
    window: tuple[int, int] | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Estimate exponential growth rate from epidemic incidence data.

    Fits log(incidence) ~ a + r * t via ordinary least squares over the
    specified window (default: full series with positive counts).

    .. math::

        I(t) = I_0 e^{r t}  \\implies  \\log I(t) = \\log I_0 + r t

    Parameters
    ----------
    incidence : array_like
        Daily (or period) incidence counts.
    window : tuple of (start, end) or None
        Index range to fit. If None, uses all time points with
        incidence > 0.
    alpha : float, default 0.05
        Significance level for CI on growth rate r.

    Returns
    -------
    dict
        Keys: 'growth_rate', 'se', 'ci_lower', 'ci_upper',
              'intercept', 'r_squared', 'doubling_time'.

    References
    ----------
    Wallinga, J. & Lipsitch, M. (2007). How generation intervals shape
    the relationship between growth rates and reproductive numbers.
    Proceedings of the Royal Society B, 274(1609), 599-604.
    """
    inc = np.asarray(incidence, dtype=float)
    if inc.ndim != 1:
        raise ValueError("incidence must be 1-D.")

    if window is not None:
        s, e = window
        t_vals = np.arange(s, e)
        y_vals = inc[s:e]
    else:
        mask = inc > 0
        t_vals = np.where(mask)[0]
        y_vals = inc[mask]

    if len(t_vals) < 3:
        raise ValueError("Need >= 3 positive-incidence time points.")

    log_y = np.log(y_vals)
    slope, intercept, r_value, p_value, se = _st.linregress(t_vals, log_y)

    z = _st.norm.ppf(1 - alpha / 2)
    ci_lo = slope - z * se
    ci_hi = slope + z * se

    dt = np.log(2) / slope if slope > 0 else np.inf

    return {
        "growth_rate": float(slope),
        "se": float(se),
        "ci_lower": float(ci_lo),
        "ci_upper": float(ci_hi),
        "intercept": float(intercept),
        "r_squared": float(r_value**2),
        "doubling_time": float(dt),
        "p_value": float(p_value),
    }


grwth = exponential_growth_rate


def cheatsheet() -> str:
    return "exponential_growth_rate({}) -> Exponential growth rate from incidence."

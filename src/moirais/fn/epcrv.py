# moirais.fn — function file (hadesllm/moirais)
"""Epidemic curve analysis (peak detection, growth rate, doubling time)."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def epidemic_curve_analysis(
    incidence: list[int] | np.ndarray,
    window: int = 7,
) -> ESRes:
    """Analyze an epidemic curve for growth dynamics.

    Unlike ``epi_c.py`` (bin counting), this function computes
    epidemiological growth metrics: peak timing, growth rate,
    doubling time, and moving averages.

    Parameters
    ----------
    incidence : array-like of int
        Daily new case counts.
    window : int, default 7
        Moving average window.

    Returns
    -------
    ESRes

    References
    ----------
    Wallinga, J. & Teunis, P. (2004). Different epidemic curves for
    severe acute respiratory syndrome reveal similar impacts of
    control measures. American Journal of Epidemiology, 160(6), 509-516.
    """
    inc = np.asarray(incidence, dtype=float)
    if len(inc) < 3:
        raise ValueError("Need at least 3 time points")

    peak_day = int(np.argmax(inc))
    peak_value = float(inc[peak_day])

    if len(inc) >= window:
        ma = np.convolve(inc, np.ones(window) / window, mode="valid")
    else:
        ma = inc.copy()

    growth_rates = []
    for i in range(1, len(inc)):
        if inc[i - 1] > 0:
            growth_rates.append(float(np.log(max(inc[i], 0.5) / inc[i - 1])))
        else:
            growth_rates.append(0.0)

    ascending = inc[:peak_day + 1] if peak_day > 0 else inc[:1]
    asc_nonzero = ascending[ascending > 0]
    if len(asc_nonzero) >= 2:
        log_inc = np.log(asc_nonzero)
        x = np.arange(len(log_inc))
        slope = float(np.polyfit(x, log_inc, 1)[0])
        doubling_time = float(np.log(2) / slope) if slope > 0 else np.inf
    else:
        slope = 0.0
        doubling_time = np.inf

    total_cases = float(np.sum(inc))

    return ESRes(
        measure="epidemic_curve_analysis",
        estimate=peak_value,
        n=len(inc),
        extra={
            "peak_day": peak_day,
            "peak_value": peak_value,
            "total_cases": total_cases,
            "growth_rate": slope,
            "doubling_time": doubling_time,
            "moving_average": ma.tolist(),
            "daily_growth_rates": growth_rates,
        },
    )


epcrv = epidemic_curve_analysis


def cheatsheet() -> str:
    return "epidemic_curve_analysis({}) -> Epidemic curve growth dynamics analysis."

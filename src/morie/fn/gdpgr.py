# morie.fn — function file (hadesllm/morie)
"""Look well into thyself; there is a source which will always spring up. — Marcus Aurelius"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def gdp_growth(
    gdp_series: np.ndarray,
) -> DescriptiveResult:
    """
    Compute GDP growth rates and trend statistics.

    .. math::

        g_t = \\frac{GDP_t - GDP_{t-1}}{GDP_{t-1}} \\times 100

    :param gdp_series: Array of GDP values over consecutive periods.
    :return: DescriptiveResult with mean growth rate and per-period rates.
    :raises ValueError: If fewer than 2 observations or any value <= 0.

    References
    ----------
    Mankiw, N. G. (2021). *Macroeconomics*. 11th ed. Worth Publishers.
    """
    gdp = np.asarray(gdp_series, dtype=np.float64)
    if len(gdp) < 2:
        raise ValueError("Need at least 2 GDP observations.")
    if np.any(gdp <= 0):
        raise ValueError("All GDP values must be positive.")

    growth_rates = np.diff(gdp) / gdp[:-1] * 100.0
    mean_growth = float(np.mean(growth_rates))
    cagr = (float((gdp[-1] / gdp[0]) ** (1.0 / (len(gdp) - 1)) - 1.0)) * 100.0

    return DescriptiveResult(
        name="GDP Growth Rate",
        value=mean_growth,
        extra={
            "growth_rates": growth_rates,
            "cagr": cagr,
            "std": float(np.std(growth_rates, ddof=1)) if len(growth_rates) > 1 else 0.0,
            "min_growth": float(np.min(growth_rates)),
            "max_growth": float(np.max(growth_rates)),
            "n_periods": len(growth_rates),
        },
    )


short = gdp_growth


def cheatsheet() -> str:
    return "gdp_growth({}) -> GDP growth rate and trend. 'Power! Unlimited power!' -- Dart"

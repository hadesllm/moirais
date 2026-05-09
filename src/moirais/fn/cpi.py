# moirais.fn — function file (hadesllm/moirais)
"""CPI and inflation rate. 'So this is how liberty dies, with thunderous applause.' -- Padme"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def cpi_inflation(
    prices_base: np.ndarray,
    prices_current: np.ndarray,
    weights: np.ndarray | None = None,
) -> DescriptiveResult:
    """
    Compute Consumer Price Index and inflation rate.

    .. math::

        CPI = \\frac{\\sum w_i \\cdot p_i^{\\text{current}}}
              {\\sum w_i \\cdot p_i^{\\text{base}}} \\times 100

    .. math::

        \\pi = CPI - 100

    :param prices_base: Array of base-period prices.
    :param prices_current: Array of current-period prices.
    :param weights: Optional consumption weights. Equal weights if None.
    :return: DescriptiveResult with CPI index and inflation rate.
    :raises ValueError: If arrays differ in length or prices are negative.

    References
    ----------
    Bureau of Labor Statistics (2018). *Handbook of Methods: Consumer
    Price Index*. U.S. Department of Labor.
    """
    p0 = np.asarray(prices_base, dtype=np.float64)
    p1 = np.asarray(prices_current, dtype=np.float64)
    if len(p0) != len(p1):
        raise ValueError("Base and current price arrays must have equal length.")
    if np.any(p0 < 0) or np.any(p1 < 0):
        raise ValueError("Prices must be non-negative.")

    if weights is None:
        w = np.ones(len(p0))
    else:
        w = np.asarray(weights, dtype=np.float64)
        if len(w) != len(p0):
            raise ValueError("Weights must have same length as prices.")

    cost_base = np.sum(w * p0)
    cost_current = np.sum(w * p1)

    if cost_base == 0:
        raise ValueError("Weighted base cost is zero; cannot compute CPI.")

    cpi_val = (cost_current / cost_base) * 100.0
    inflation = cpi_val - 100.0

    return DescriptiveResult(
        name="Consumer Price Index",
        value=float(cpi_val),
        extra={
            "inflation_rate": float(inflation),
            "cost_base": float(cost_base),
            "cost_current": float(cost_current),
            "n_items": len(p0),
        },
    )


short = cpi_inflation


def cheatsheet() -> str:
    return "cpi_inflation({}) -> CPI and inflation rate. 'So this is how liberty dies, with t"

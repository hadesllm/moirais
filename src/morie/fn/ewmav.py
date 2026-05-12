# morie.fn — function file (hadesllm/morie)
"""EWMA volatility (RiskMetrics)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def ewma_volatility(
    returns: np.ndarray,
    lambda_: float = 0.94,
) -> DescriptiveResult:
    r"""Exponentially weighted moving average volatility.

    The RiskMetrics variance estimator:

    .. math::

        \\sigma_t^2 = \\lambda \\, \\sigma_{t-1}^2
                    + (1 - \\lambda) \\, r_{t-1}^2

    Parameters
    ----------
    returns : array-like
        Array of returns.
    lambda\\_ : float, default 0.94
        Decay factor in (0, 1).  JP Morgan RiskMetrics uses 0.94
        for daily data and 0.97 for monthly.

    Returns
    -------
    DescriptiveResult
        ``value`` is the latest EWMA volatility (standard deviation).
        ``extra`` has ``variance_series`` and ``volatility_series``.

    Raises
    ------
    ValueError
        If fewer than 2 returns or lambda out of range.

    References
    ----------
    JP Morgan / Reuters (1996). *RiskMetrics -- Technical Document*
    (4th ed.). New York.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    if len(r) < 2:
        raise ValueError("Need at least 2 return observations.")
    if not 0 < lambda_ < 1:
        raise ValueError(f"lambda_ must be in (0, 1), got {lambda_}.")

    var_series = np.zeros(len(r))
    var_series[0] = r[0] ** 2

    for t in range(1, len(r)):
        var_series[t] = lambda_ * var_series[t - 1] + (1 - lambda_) * r[t - 1] ** 2

    vol_series = np.sqrt(var_series)

    return DescriptiveResult(
        name="EWMAVolatility",
        value=float(vol_series[-1]),
        extra={
            "variance_series": var_series,
            "volatility_series": vol_series,
            "lambda": lambda_,
            "n_periods": len(r),
        },
    )


ewmav = ewma_volatility


def cheatsheet() -> str:
    return "ewma_volatility({}) -> EWMA volatility (RiskMetrics)."

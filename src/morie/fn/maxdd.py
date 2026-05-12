# morie.fn -- function file (hadesllm/morie)
"""Maximum drawdown."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def max_drawdown(prices: np.ndarray) -> DescriptiveResult:
    r"""Maximum drawdown from a price series.

    .. math::

        \\text{MDD} = \\max_{t \\in [0,T]}
        \\left(\\frac{\\max_{s \\in [0,t]} P_s - P_t}
        {\\max_{s \\in [0,t]} P_s}\\right)

    Parameters
    ----------
    prices : array-like
        Time series of asset prices (positive values).

    Returns
    -------
    DescriptiveResult
        ``value`` is the maximum drawdown (as a positive fraction,
        e.g. 0.25 = 25%).  ``extra`` has ``peak_idx``, ``trough_idx``,
        ``peak_price``, ``trough_price``, ``drawdown_series``.

    Raises
    ------
    ValueError
        If fewer than 2 prices or any price is non-positive.

    References
    ----------
    Magdon-Ismail, M., & Atiya, A. F. (2004). Maximum drawdown.
    *Risk Magazine*, 17(10), 99--102.
    """
    p = np.asarray(prices, dtype=np.float64).ravel()
    if len(p) < 2:
        raise ValueError("Need at least 2 price observations.")
    if np.any(p <= 0):
        raise ValueError("All prices must be positive.")

    running_max = np.maximum.accumulate(p)
    drawdowns = (running_max - p) / running_max

    mdd = float(np.max(drawdowns))
    trough_idx = int(np.argmax(drawdowns))
    peak_idx = int(np.argmax(p[: trough_idx + 1]))

    return DescriptiveResult(
        name="MaxDrawdown",
        value=mdd,
        extra={
            "peak_idx": peak_idx,
            "trough_idx": trough_idx,
            "peak_price": float(p[peak_idx]),
            "trough_price": float(p[trough_idx]),
            "drawdown_series": drawdowns,
        },
    )


maxdd = max_drawdown


def cheatsheet() -> str:
    return "max_drawdown({}) -> Maximum drawdown."

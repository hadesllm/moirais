"""Sharpe ratio."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def sharpe_ratio(
    returns: np.ndarray,
    risk_free: float = 0.0,
) -> DescriptiveResult:
    r"""Annualized Sharpe ratio.

    .. math::

        S = \\frac{\\bar{R} - R_f}{\\sigma_R}

    where :math:`\\bar{R}` is the mean return, :math:`R_f` is the
    risk-free rate (per period), and :math:`\\sigma_R` is the standard
    deviation of returns.

    Parameters
    ----------
    returns : array-like
        Array of periodic returns (e.g., daily log-returns).
    risk_free : float, default 0.0
        Risk-free rate per period.

    Returns
    -------
    DescriptiveResult
        ``value`` is the Sharpe ratio.  ``extra`` has ``mean_excess``,
        ``volatility``, ``n_periods``.

    Raises
    ------
    ValueError
        If fewer than 2 returns or zero volatility.

    References
    ----------
    Sharpe, W. F. (1966). Mutual fund performance.
    *Journal of Business*, 39(1), 119--138.

    Sharpe, W. F. (1994). The Sharpe ratio.
    *Journal of Portfolio Management*, 21(1), 49--58.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    if len(r) < 2:
        raise ValueError("Need at least 2 return observations.")

    excess = r - risk_free
    vol = float(np.std(excess, ddof=1))
    if vol == 0:
        raise ValueError("Zero volatility; Sharpe ratio undefined.")

    mean_excess = float(np.mean(excess))
    sr = mean_excess / vol

    return DescriptiveResult(
        name="SharpeRatio",
        value=float(sr),
        extra={
            "mean_excess": mean_excess,
            "volatility": vol,
            "n_periods": len(r),
            "risk_free": risk_free,
        },
    )


shrpe = sharpe_ratio


def cheatsheet() -> str:
    return "sharpe_ratio({}) -> Sharpe ratio."

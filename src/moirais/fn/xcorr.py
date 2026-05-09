"""Cross-correlation between two signals."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def cross_correlation(x, y, max_lag: int | None = None) -> DescriptiveResult:
    """Compute normalised cross-correlation between *x* and *y*.

    Parameters
    ----------
    x, y : array-like
        Input signals.
    max_lag : int or None
        Maximum lag to compute. Default uses len(x)-1.

    Returns
    -------
    DescriptiveResult
    """
    from moirais._filters import cross_correlation as _xc

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    corr = _xc(x, y, max_lag=max_lag)
    return DescriptiveResult(
        name="cross_correlation",
        value=float(np.max(np.abs(corr))),
        extra={"correlation": corr, "max_lag": max_lag},
    )


xcorr = cross_correlation


def cheatsheet() -> str:
    return "cross_correlation({}) -> Cross-correlation between two signals."

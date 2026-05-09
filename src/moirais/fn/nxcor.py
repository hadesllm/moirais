# moirais.fn — function file (hadesllm/moirais)
"""Normalized cross-correlation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The more you know, the more you realize you don't know. — Aristotle"


def normalized_xcorr(x, y, **kwargs) -> DescriptiveResult:
    """Compute the normalized cross-correlation between *x* and *y*.

    Returns the peak value and the lag at which it occurs.

    Parameters
    ----------
    x, y : array-like
        Input signals.

    Returns
    -------
    DescriptiveResult
        ``value`` is the peak normalized cross-correlation.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - np.mean(x)
    y = y - np.mean(y)
    norm = np.sqrt(np.sum(x**2) * np.sum(y**2))
    if norm == 0.0:
        return DescriptiveResult(
            name="normalized_xcorr",
            value=0.0,
            extra={"peak": 0.0, "lag": 0, "n": len(x)},
        )
    xcorr = np.correlate(x, y, mode="full")
    xcorr_norm = xcorr / norm
    peak_idx = int(np.argmax(np.abs(xcorr_norm)))
    lag = peak_idx - (len(y) - 1)
    peak = float(xcorr_norm[peak_idx])
    return DescriptiveResult(
        name="normalized_xcorr",
        value=peak,
        extra={"peak": peak, "lag": lag, "xcorr": xcorr_norm, "n": len(x)},
    )


nxcor = normalized_xcorr


def cheatsheet() -> str:
    return "normalized_xcorr({}) -> Normalized cross-correlation."

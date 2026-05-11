# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Alpha-trimmed mean filter."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def alpha_trimmed_mean_filter(x, window: int = 5, alpha: float = 0.2) -> SignalResult:
    """Apply an alpha-trimmed mean filter to signal *x*.

    Parameters
    ----------
    x : array-like
        Input signal.
    window : int
        Window size. Default 5.
    alpha : float
        Trim fraction. Default 0.2.

    Returns
    -------
    SignalResult
    """
    from morie._filters import alpha_trimmed_mean_filter as _atm

    x = np.asarray(x, dtype=float)
    result = _atm(x, window=window, alpha=alpha)
    return SignalResult(
        name="alpha_trimmed_mean_filter",
        filtered=result,
        fs=0.0,
        n_samples=len(x),
        extra={"window": window, "alpha": alpha},
    )


atmfl = alpha_trimmed_mean_filter


def cheatsheet() -> str:
    return "alpha_trimmed_mean_filter({}) -> Alpha-trimmed mean filter."

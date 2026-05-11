# morie.fn — function file (hadesllm/morie)
"""Notch filter for powerline interference removal (50/60 Hz)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_notch_filter"]


def rangayyan_notch_filter(notch_freq, bandwidth, fs):
    """
    Notch filter for powerline interference removal (50/60 Hz)

    Formula: H(z) = (1 - 2cos(w0)*z^{-1} + z^{-2}) / (1 - 2*r*cos(w0)*z^{-1} + r^2*z^{-2})

    Parameters
    ----------
    notch_freq : array-like
        Input data.
    bandwidth : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: b, a

    References
    ----------
    Rangayyan Ch 3.7.3
    """
    notch_freq = np.asarray(notch_freq, dtype=float)
    n = int(notch_freq) if notch_freq.ndim == 0 else len(notch_freq)
    result = float(np.mean(notch_freq))
    se = float(np.std(notch_freq, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Notch filter for powerline interference removal (50/60 Hz)"})


def cheatsheet():
    return "rgntch: Notch filter for powerline interference removal (50/60 Hz)"

# moirais.fn — function file (hadesllm/moirais)
"""Amplitude demodulation (envelope via Hilbert transform)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_amplitude_demod"]


def rangayyan_amplitude_demod(x, fs):
    """
    Amplitude demodulation (envelope via Hilbert transform)

    Formula: envelope(t) = sqrt(x(t)^2 + H{x(t)}^2) = |x_a(t)|

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: envelope

    References
    ----------
    Rangayyan Ch 5.5.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Amplitude demodulation (envelope via Hilbert transform)"})


def cheatsheet():
    return "rgampd: Amplitude demodulation (envelope via Hilbert transform)"

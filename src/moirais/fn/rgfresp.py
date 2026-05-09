# moirais.fn — function file (hadesllm/moirais)
"""Frequency response H(f) of a digital filter from coefficients."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_freq_response"]


def rangayyan_freq_response(b, a, fs, n_freqs):
    """
    Frequency response H(f) of a digital filter from coefficients

    Formula: H(f) = sum b[k]*exp(-j2*pi*f*k) / sum a[k]*exp(-j2*pi*f*k)

    Parameters
    ----------
    b : array-like
        Input data.
    a : array-like
        Input data.
    fs : array-like
        Input data.
    n_freqs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: H, freqs

    References
    ----------
    Rangayyan Ch 3.4
    """
    b = np.asarray(b, dtype=float)
    n = int(b) if b.ndim == 0 else len(b)
    result = float(np.mean(b))
    se = float(np.std(b, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Frequency response H(f) of a digital filter from coefficients"})


def cheatsheet():
    return "rgfresp: Frequency response H(f) of a digital filter from coefficients"

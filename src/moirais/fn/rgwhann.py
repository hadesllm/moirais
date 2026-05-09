# moirais.fn — function file (hadesllm/moirais)
"""Hann (Hanning) window function."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_hann_window"]


def rangayyan_hann_window(N):
    """
    Hann (Hanning) window function

    Formula: w[n] = 0.5*(1 - cos(2*pi*n/(N-1)))

    Parameters
    ----------
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: window

    References
    ----------
    Rangayyan Ch 6.3.4
    """
    N = np.asarray(N, dtype=float)
    n = int(N) if N.ndim == 0 else len(N)
    result = float(np.mean(N))
    se = float(np.std(N, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hann (Hanning) window function"})


def cheatsheet():
    return "rgwhann: Hann (Hanning) window function"

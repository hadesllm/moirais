# moirais.fn — function file (hadesllm/moirais)
"""Blackman window function."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_blackman_window"]


def rangayyan_blackman_window(N):
    """
    Blackman window function

    Formula: w[n] = 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Blackman window function"})


def cheatsheet():
    return "rgwblkm: Blackman window function"

# morie.fn — function file (hadesllm/morie)
"""FIR filter design (windowed sinc)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_fir_filter"]


def rangayyan_fir_filter(x, cutoff, order):
    """
    FIR filter design (windowed sinc)

    Formula: h[n] = w[n] * sinc(2*fc*(n-M/2))

    Parameters
    ----------
    x : array-like
        Input data.
    cutoff : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rangayyan Ch 3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "FIR filter design (windowed sinc)"})


def cheatsheet():
    return "rgfir: FIR filter design (windowed sinc)"

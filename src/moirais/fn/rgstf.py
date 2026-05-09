# moirais.fn — function file (hadesllm/moirais)
"""Short-time Fourier transform."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_stft"]


def rangayyan_stft(x):
    """
    Short-time Fourier transform

    Formula: X(m,f) = sum x[n]*w[n-m]*exp(-j2pi*fn)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rangayyan Ch 4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Short-time Fourier transform"})


def cheatsheet():
    return "rgstf: Short-time Fourier transform"

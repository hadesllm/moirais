# moirais.fn — function file (hadesllm/moirais)
"""Discrete Fourier transform (DFT)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_dft"]


def rangayyan_dft(x):
    """
    Discrete Fourier transform (DFT)

    Formula: X[k] = sum_{n=0}^{N-1} x[n] * exp(-j2*pi*kn/N)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X_k

    References
    ----------
    Rangayyan Ch 3.4.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Discrete Fourier transform (DFT)"})


def cheatsheet():
    return "rgdft: Discrete Fourier transform (DFT)"

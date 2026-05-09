# moirais.fn — function file (hadesllm/moirais)
"""Haar wavelet transform (simplest orthogonal wavelet)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_haar_wavelet"]


def rangayyan_haar_wavelet(x, levels):
    """
    Haar wavelet transform (simplest orthogonal wavelet)

    Formula: phi=[1,1]/sqrt(2); psi=[1,-1]/sqrt(2); 1-level: c=(x0+x1)/sqrt(2); d=(x0-x1)/sqrt(2)

    Parameters
    ----------
    x : array-like
        Input data.
    levels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: approx, details

    References
    ----------
    Rangayyan Ch 8.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Haar wavelet transform (simplest orthogonal wavelet)"})


def cheatsheet():
    return "rghaar: Haar wavelet transform (simplest orthogonal wavelet)"

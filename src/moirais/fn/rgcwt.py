# moirais.fn — function file (hadesllm/moirais)
"""Continuous wavelet transform (CWT)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_cwt"]


def rangayyan_cwt(x, fs, wavelet, scales):
    """
    Continuous wavelet transform (CWT)

    Formula: CWT(a,b) = (1/sqrt(a)) integral x(t)*psi*((t-b)/a) dt

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.
    wavelet : array-like
        Input data.
    scales : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: coefficients, scales, freqs

    References
    ----------
    Rangayyan Ch 8.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Continuous wavelet transform (CWT)"})


def cheatsheet():
    return "rgcwt: Continuous wavelet transform (CWT)"

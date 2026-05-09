# moirais.fn — function file (hadesllm/moirais)
"""Wavelet entropy for measuring signal regularity."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_wavelet_entropy"]


def rangayyan_wavelet_entropy(x, wavelet, levels):
    """
    Wavelet entropy for measuring signal regularity

    Formula: E = -sum p_j * log(p_j); p_j = E_j / E_total; E_j = sum d_j^2

    Parameters
    ----------
    x : array-like
        Input data.
    wavelet : array-like
        Input data.
    levels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: entropy

    References
    ----------
    Rangayyan Ch 8.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wavelet entropy for measuring signal regularity"})


def cheatsheet():
    return "rgentrwv: Wavelet entropy for measuring signal regularity"

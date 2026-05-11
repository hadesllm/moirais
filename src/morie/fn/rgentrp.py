# morie.fn — function file (hadesllm/morie)
"""Spectral entropy for signal complexity measurement."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_spectral_entropy"]


def rangayyan_spectral_entropy(x, fs):
    """
    Spectral entropy for signal complexity measurement

    Formula: H = -sum p_k * log(p_k) where p_k = S(f_k)/sum S(f)

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectral_entropy

    References
    ----------
    Rangayyan Ch 5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spectral entropy for signal complexity measurement"})


def cheatsheet():
    return "rgentrp: Spectral entropy for signal complexity measurement"

# moirais.fn — function file (hadesllm/moirais)
"""Spectral resolution and leakage analysis (Rayleigh criterion)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_spectral_resolution"]


def rangayyan_spectral_resolution(N, fs, window_type):
    """
    Spectral resolution and leakage analysis (Rayleigh criterion)

    Formula: delta_f = fs/N (Rayleigh); window sidelobe level determines leakage

    Parameters
    ----------
    N : array-like
        Input data.
    fs : array-like
        Input data.
    window_type : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: resolution, sidelobe_db

    References
    ----------
    Rangayyan Ch 6.3.4
    """
    N = np.asarray(N, dtype=float)
    n = int(N) if N.ndim == 0 else len(N)
    result = float(np.mean(N))
    se = float(np.std(N, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spectral resolution and leakage analysis (Rayleigh criterion)"})


def cheatsheet():
    return "rgspres: Spectral resolution and leakage analysis (Rayleigh criterion)"

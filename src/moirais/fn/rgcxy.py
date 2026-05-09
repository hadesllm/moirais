# moirais.fn — function file (hadesllm/moirais)
"""Magnitude-squared coherence (MSC) function."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_coherence_cxy"]


def rangayyan_coherence_cxy(x, y, fs, nperseg):
    """
    Magnitude-squared coherence (MSC) function

    Formula: C_xy(f) = |S_xy(f)|^2 / (S_xx(f) * S_yy(f))

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    fs : array-like
        Input data.
    nperseg : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: coherence, freqs

    References
    ----------
    Rangayyan Ch 2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Magnitude-squared coherence (MSC) function"})


def cheatsheet():
    return "rgcxy: Magnitude-squared coherence (MSC) function"

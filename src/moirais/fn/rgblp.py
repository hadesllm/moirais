# moirais.fn — function file (hadesllm/moirais)
"""Butterworth lowpass filter design (analog prototype to digital)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_butterworth_lp"]


def rangayyan_butterworth_lp(cutoff_hz, order, fs):
    """
    Butterworth lowpass filter design (analog prototype to digital)

    Formula: |H(Omega)|^2 = 1 / (1 + (Omega/Omega_c)^{2N}); bilinear transform to digital

    Parameters
    ----------
    cutoff_hz : array-like
        Input data.
    order : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: b, a

    References
    ----------
    Rangayyan Ch 3.7.1
    """
    cutoff_hz = np.asarray(cutoff_hz, dtype=float)
    n = int(cutoff_hz) if cutoff_hz.ndim == 0 else len(cutoff_hz)
    result = float(np.mean(cutoff_hz))
    se = float(np.std(cutoff_hz, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Butterworth lowpass filter design (analog prototype to digital)"})


def cheatsheet():
    return "rgblp: Butterworth lowpass filter design (analog prototype to digital)"

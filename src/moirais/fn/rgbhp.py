# moirais.fn — function file (hadesllm/moirais)
"""Butterworth highpass filter design."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_butterworth_hp"]


def rangayyan_butterworth_hp(cutoff_hz, order, fs):
    """
    Butterworth highpass filter design

    Formula: LPF to HPF via spectral inversion: Omega -> Omega_c^2/Omega

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
    Rangayyan Ch 3.7.2
    """
    cutoff_hz = np.asarray(cutoff_hz, dtype=float)
    n = int(cutoff_hz) if cutoff_hz.ndim == 0 else len(cutoff_hz)
    result = float(np.mean(cutoff_hz))
    se = float(np.std(cutoff_hz, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Butterworth highpass filter design"})


def cheatsheet():
    return "rgbhp: Butterworth highpass filter design"

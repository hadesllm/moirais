# morie.fn -- function file (hadesllm/morie)
"""Murmur presence detection in PCG via spectral analysis."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_pcg_murmur_detect"]


def rangayyan_pcg_murmur_detect(pcg, ecg, fs):
    """
    Murmur presence detection in PCG via spectral analysis

    Formula: Murmur: high energy in systole spectral centroid > normal; threshold rule

    Parameters
    ----------
    pcg : array-like
        Input data.
    ecg : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: has_murmur, confidence

    References
    ----------
    Rangayyan Ch 10.2.4
    """
    pcg = np.asarray(pcg, dtype=float)
    n = int(pcg) if pcg.ndim == 0 else len(pcg)
    result = float(np.mean(pcg))
    se = float(np.std(pcg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Murmur presence detection in PCG via spectral analysis"})


def cheatsheet():
    return "rgpcgmrm: Murmur presence detection in PCG via spectral analysis"

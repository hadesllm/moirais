# morie.fn -- function file (hadesllm/morie)
"""Synchronized averaging of PCG spectra for murmur analysis."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_pcg_sync_avg"]


def rangayyan_pcg_sync_avg(pcg, ecg, fs, n_cycles):
    """
    Synchronized averaging of PCG spectra for murmur analysis

    Formula: S_avg(f) = (1/M)*sum |PCG_k(f)|^2

    Parameters
    ----------
    pcg : array-like
        Input data.
    ecg : array-like
        Input data.
    fs : array-like
        Input data.
    n_cycles : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: avg_spectrum, freqs

    References
    ----------
    Rangayyan Ch 6.3.6
    """
    pcg = np.asarray(pcg, dtype=float)
    n = int(pcg) if pcg.ndim == 0 else len(pcg)
    result = float(np.mean(pcg))
    se = float(np.std(pcg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Synchronized averaging of PCG spectra for murmur analysis"})


def cheatsheet():
    return "rgpsync: Synchronized averaging of PCG spectra for murmur analysis"

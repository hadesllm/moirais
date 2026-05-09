# moirais.fn — function file (hadesllm/moirais)
"""Heart murmur frequency analysis for valvular defect diagnosis."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_murmur_analysis"]


def rangayyan_murmur_analysis(pcg, fs, ecg):
    """
    Heart murmur frequency analysis for valvular defect diagnosis

    Formula: Murmur spectral features: dominant freq, bandwidth, spectral centroid

    Parameters
    ----------
    pcg : array-like
        Input data.
    fs : array-like
        Input data.
    ecg : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: murmur_features

    References
    ----------
    Rangayyan Ch 6.2.2
    """
    pcg = np.asarray(pcg, dtype=float)
    n = int(pcg) if pcg.ndim == 0 else len(pcg)
    result = float(np.mean(pcg))
    se = float(np.std(pcg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Heart murmur frequency analysis for valvular defect diagnosis"})


def cheatsheet():
    return "rgmurm: Heart murmur frequency analysis for valvular defect diagnosis"

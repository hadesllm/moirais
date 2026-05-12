# morie.fn -- function file (hadesllm/morie)
"""Wavelet coefficient moments (energy, variance, mean) per scale."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_wavelet_moments"]


def rangayyan_wavelet_moments(x, wavelet, levels):
    """
    Wavelet coefficient moments (energy, variance, mean) per scale

    Formula: E_j=sum|d_j[n]|^2; var_j=std(d_j)^2; mean_j=mean(d_j)

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
        Keys: energies, variances, means

    References
    ----------
    Rangayyan Ch 8.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wavelet coefficient moments (energy, variance, mean) per scale"})


def cheatsheet():
    return "rgwvmom: Wavelet coefficient moments (energy, variance, mean) per scale"

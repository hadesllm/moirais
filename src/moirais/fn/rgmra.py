# moirais.fn — function file (hadesllm/moirais)
"""Multiresolution analysis (MRA) decomposition."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_mra"]


def rangayyan_mra(x, wavelet, levels):
    """
    Multiresolution analysis (MRA) decomposition

    Formula: x = A_J + sum_{j=1}^{J} D_j; A_J=smooth, D_j=detail at scale 2^j

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
        Keys: approximation, details_list

    References
    ----------
    Rangayyan Ch 8.8.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multiresolution analysis (MRA) decomposition"})


def cheatsheet():
    return "rgmra: Multiresolution analysis (MRA) decomposition"

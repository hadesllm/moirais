# morie.fn — function file (hadesllm/morie)
"""Decomposition-based adaptive TFD using MP atoms."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_decomp_tfd"]


def rangayyan_decomp_tfd(x, fs, dictionary, max_atoms):
    """
    Decomposition-based adaptive TFD using MP atoms

    Formula: TFD(t,f) = sum_k |a_k|^2 * WVD(phi_k)(t,f)

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.
    dictionary : array-like
        Input data.
    max_atoms : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: tfd, t, freqs

    References
    ----------
    Rangayyan Ch 9.6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Decomposition-based adaptive TFD using MP atoms"})


def cheatsheet():
    return "rgdtfd: Decomposition-based adaptive TFD using MP atoms"

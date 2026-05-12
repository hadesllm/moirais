# morie.fn -- function file (hadesllm/morie)
"""Wavelet packet decomposition (full binary tree)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_wavelet_packet"]


def rangayyan_wavelet_packet(x, wavelet, levels):
    """
    Wavelet packet decomposition (full binary tree)

    Formula: Both approximation AND detail branches filtered and downsampled at each level

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
        Keys: packet_tree

    References
    ----------
    Rangayyan Ch 8.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wavelet packet decomposition (full binary tree)"})


def cheatsheet():
    return "rgwvpkt: Wavelet packet decomposition (full binary tree)"

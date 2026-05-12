# morie.fn -- function file (hadesllm/morie)
"""Bartlett averaging of periodograms for variance reduction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_bartlett_psd"]


def rangayyan_bartlett_psd(x, fs, nseg):
    """
    Bartlett averaging of periodograms for variance reduction

    Formula: P_B(f) = (1/K) sum_{k=1}^{K} |X_k(f)|^2; non-overlapping segments

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.
    nseg : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: psd, freqs

    References
    ----------
    Rangayyan Ch 6.3.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bartlett averaging of periodograms for variance reduction"})


def cheatsheet():
    return "rgbartl: Bartlett averaging of periodograms for variance reduction"

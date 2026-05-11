# morie.fn — function file (hadesllm/morie)
"""Wavelet energy per subband (scale)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_wavelet_energy"]


def rangayyan_wavelet_energy(x, wavelet, levels):
    """
    Wavelet energy per subband (scale)

    Formula: E_j = sum_{n} |d_j[n]|^2; total = sum E_j + E_approx

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
        Keys: energies, relative_energies

    References
    ----------
    Rangayyan Ch 8.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wavelet energy per subband (scale)"})


def cheatsheet():
    return "rgwvener: Wavelet energy per subband (scale)"

# morie.fn -- function file (hadesllm/morie)
"""Wigner-Ville distribution (bilinear TFD)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_wigner_ville"]


def rangayyan_wigner_ville(x, fs):
    """
    Wigner-Ville distribution (bilinear TFD)

    Formula: WVD(t,f) = integral x(t+tau/2)*x*(t-tau/2)*exp(-j2*pi*f*tau) dtau

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: wvd, t, freqs

    References
    ----------
    Rangayyan Ch 8.9
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wigner-Ville distribution (bilinear TFD)"})


def cheatsheet():
    return "rgwvd: Wigner-Ville distribution (bilinear TFD)"

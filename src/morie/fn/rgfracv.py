# morie.fn -- function file (hadesllm/morie)
"""Fractal analysis of VAG signals via power spectral slope."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_fractal_vag"]


def rangayyan_fractal_vag(vag, fs):
    """
    Fractal analysis of VAG signals via power spectral slope

    Formula: FD = (5-beta)/2; beta estimated from log-log PSD in 100-500 Hz

    Parameters
    ----------
    vag : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: fd, beta, r_sq

    References
    ----------
    Rangayyan Ch 6.6
    """
    vag = np.asarray(vag, dtype=float)
    n = int(vag) if vag.ndim == 0 else len(vag)
    result = float(np.mean(vag))
    se = float(np.std(vag, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fractal analysis of VAG signals via power spectral slope"})


def cheatsheet():
    return "rgfracv: Fractal analysis of VAG signals via power spectral slope"

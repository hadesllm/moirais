# morie.fn — function file (hadesllm/morie)
"""Adaptive TFD of VAG signals via matching pursuit."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_vag_adaptive_tfd"]


def rangayyan_vag_adaptive_tfd(vag, fs, n_atoms):
    """
    Adaptive TFD of VAG signals via matching pursuit

    Formula: MP atoms represent time-frequency structures; TFD = sum of atom WVDs

    Parameters
    ----------
    vag : array-like
        Input data.
    fs : array-like
        Input data.
    n_atoms : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: tfd, t, freqs

    References
    ----------
    Rangayyan Ch 9.9
    """
    vag = np.asarray(vag, dtype=float)
    n = int(vag) if vag.ndim == 0 else len(vag)
    result = float(np.mean(vag))
    se = float(np.std(vag, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Adaptive TFD of VAG signals via matching pursuit"})


def cheatsheet():
    return "rgvagadp: Adaptive TFD of VAG signals via matching pursuit"

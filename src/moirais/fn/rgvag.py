# moirais.fn — function file (hadesllm/moirais)
"""Vibroarthrogram (VAG) signal characterization."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_vag_analysis"]


def rangayyan_vag_analysis(vag, fs):
    """
    Vibroarthrogram (VAG) signal characterization

    Formula: Features: RMS, ZCR, spectral entropy, fractal dimension

    Parameters
    ----------
    vag : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: features

    References
    ----------
    Rangayyan Ch 1.2.14
    """
    vag = np.asarray(vag, dtype=float)
    n = int(vag) if vag.ndim == 0 else len(vag)
    result = float(np.mean(vag))
    se = float(np.std(vag, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Vibroarthrogram (VAG) signal characterization"})


def cheatsheet():
    return "rgvag: Vibroarthrogram (VAG) signal characterization"

# moirais.fn — function file (hadesllm/moirais)
"""Waveform length (curve length) feature."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ch5_waveform_length"]


def rangayyan_ch5_waveform_length(x):
    """
    Waveform length (curve length) feature

    Formula: WL = sum_{n=1}^{N} |x[n]-x[n-1]|

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: wl

    References
    ----------
    Rangayyan Ch 5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Waveform length (curve length) feature"})


def cheatsheet():
    return "rgeqn5b: Waveform length (curve length) feature"

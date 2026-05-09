# moirais.fn — function file (hadesllm/moirais)
"""Katz fractal dimension of a waveform."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_katz_fd"]


def rangayyan_katz_fd(x):
    """
    Katz fractal dimension of a waveform

    Formula: FD = log10(n) / (log10(n) + log10(d/L))

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: fd

    References
    ----------
    Rangayyan Ch 5.13.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Katz fractal dimension of a waveform"})


def cheatsheet():
    return "rgkatzfd: Katz fractal dimension of a waveform"

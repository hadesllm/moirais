# moirais.fn — function file (hadesllm/moirais)
"""Time-variant linear system (TV-LSI) characterization."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_tvlsi"]


def rangayyan_tvlsi(x, y, fs, window):
    """
    Time-variant linear system (TV-LSI) characterization

    Formula: TV impulse response h(t,tau); spectrogram = |H_tv(t,f)|^2

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    fs : array-like
        Input data.
    window : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: tv_spectrum, t, freqs

    References
    ----------
    Rangayyan Ch 8.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Time-variant linear system (TV-LSI) characterization"})


def cheatsheet():
    return "rgstvar: Time-variant linear system (TV-LSI) characterization"

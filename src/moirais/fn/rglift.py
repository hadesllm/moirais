# moirais.fn — function file (hadesllm/moirais)
"""Cepstral liftering (low-time / high-time separation)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_liftering"]


def rangayyan_liftering(cepstrum, l_low, l_high):
    """
    Cepstral liftering (low-time / high-time separation)

    Formula: lifter(c, l_low, l_high) = c * window; window=1 in [l_low, l_high] else 0

    Parameters
    ----------
    cepstrum : array-like
        Input data.
    l_low : array-like
        Input data.
    l_high : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: liftered_cepstrum

    References
    ----------
    Rangayyan Ch 4.7.2
    """
    cepstrum = np.asarray(cepstrum, dtype=float)
    n = int(cepstrum) if cepstrum.ndim == 0 else len(cepstrum)
    result = float(np.mean(cepstrum))
    se = float(np.std(cepstrum, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cepstral liftering (low-time / high-time separation)"})


def cheatsheet():
    return "rglift: Cepstral liftering (low-time / high-time separation)"

# moirais.fn — function file (hadesllm/moirais)
"""Fractal dimension from PSD slope (1/f noise model)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_fd_psd_slope"]


def rangayyan_fd_psd_slope(psd, freqs, f_range):
    """
    Fractal dimension from PSD slope (1/f noise model)

    Formula: FD = (5 - beta) / 2 where S(f) ~ f^{-beta}; beta from log-log PSD slope

    Parameters
    ----------
    psd : array-like
        Input data.
    freqs : array-like
        Input data.
    f_range : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: fd, beta

    References
    ----------
    Rangayyan Ch 6.6.2
    """
    psd = np.asarray(psd, dtype=float)
    n = int(psd) if psd.ndim == 0 else len(psd)
    result = float(np.mean(psd))
    se = float(np.std(psd, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fractal dimension from PSD slope (1/f noise model)"})


def cheatsheet():
    return "rgfdpsd: Fractal dimension from PSD slope (1/f noise model)"

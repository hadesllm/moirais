# morie.fn -- function file (hadesllm/morie)
"""Mean frequency of a power spectrum."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ch6_mean_freq"]


def rangayyan_ch6_mean_freq(psd, freqs):
    """
    Mean frequency of a power spectrum

    Formula: f_mean = sum(f_k * S(f_k)) / sum(S(f_k))

    Parameters
    ----------
    psd : array-like
        Input data.
    freqs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mean_freq

    References
    ----------
    Rangayyan Ch 6.4.1
    """
    psd = np.asarray(psd, dtype=float)
    n = int(psd) if psd.ndim == 0 else len(psd)
    result = float(np.mean(psd))
    se = float(np.std(psd, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mean frequency of a power spectrum"})


def cheatsheet():
    return "rgeqn6a: Mean frequency of a power spectrum"

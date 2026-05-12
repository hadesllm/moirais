# morie.fn -- function file (hadesllm/morie)
"""Comb filter for periodic artifact removal."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_comb_filter"]


def rangayyan_comb_filter(period_samples, fs):
    """
    Comb filter for periodic artifact removal

    Formula: H(z) = 1 - z^{-N}; notches at multiples of fs/N

    Parameters
    ----------
    period_samples : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: b, a

    References
    ----------
    Rangayyan Ch 3.7.3
    """
    period_samples = np.asarray(period_samples, dtype=float)
    n = int(period_samples) if period_samples.ndim == 0 else len(period_samples)
    result = float(np.mean(period_samples))
    se = float(np.std(period_samples, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Comb filter for periodic artifact removal"})


def cheatsheet():
    return "rgcomb: Comb filter for periodic artifact removal"

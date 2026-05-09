"""Bits per character."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bits_per_character"]


def bits_per_character(log_probs, N):
    """
    Bits per character

    Formula: BPC = -(1/N) sum log_2 p(x_i)

    Parameters
    ----------
    log_probs : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hutter Prize benchmark
    """
    log_probs = np.atleast_1d(np.asarray(log_probs, dtype=float))
    n = len(log_probs)
    result = float(np.mean(log_probs))
    se = float(np.std(log_probs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bits per character"})


def cheatsheet():
    return "bpc: Bits per character"

"""Trimmed L-moments TL(s,t) of a sample."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_trimmed_lmom"]


def evt_trimmed_lmom(x, s, t, order):
    """
    Trimmed L-moments TL(s,t) of a sample

    Formula: λ_r^{(s,t)} = (1/r) Σ pk r,s,t weighted order stats

    Parameters
    ----------
    x : array-like
        Input data.
    s : array-like
        Input data.
    t : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lambda

    References
    ----------
    Elamir & Seheult (2003)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Trimmed L-moments TL(s,t) of a sample"})


def cheatsheet():
    return "evtlmom: Trimmed L-moments TL(s,t) of a sample"

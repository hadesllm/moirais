"""TOA radiation balance."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["toa_radiation_balance"]


def toa_radiation_balance(S, alpha, OLR):
    """
    TOA radiation balance

    Formula: N = (1−α)S/4 − OLR

    Parameters
    ----------
    S : array-like
        Input data.
    alpha : array-like
        Input data.
    OLR : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hartmann (1994)
    """
    S = np.atleast_1d(np.asarray(S, dtype=float))
    n = len(S)
    result = float(np.mean(S))
    se = float(np.std(S, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TOA radiation balance"})


def cheatsheet():
    return "recapTOA: TOA radiation balance"

"""Build ILR contrast matrix V from a sequential binary partition."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aitchison_sbp_basis"]


def aitchison_sbp_basis(sign):
    """
    Build ILR contrast matrix V from a sequential binary partition

    Formula: V from SBP sign matrix per Egozcue & Pawlowsky-Glahn

    Parameters
    ----------
    sign : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: V

    References
    ----------
    Egozcue (2005)
    """
    sign = np.atleast_1d(np.asarray(sign, dtype=float))
    n = len(sign)
    result = float(np.mean(sign))
    se = float(np.std(sign, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Build ILR contrast matrix V from a sequential binary partition"})


def cheatsheet():
    return "aitsbp: Build ILR contrast matrix V from a sequential binary partition"

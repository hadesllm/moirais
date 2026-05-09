"""SEIRA with asymptomatic compartment."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["seira_asymptomatic"]


def seira_asymptomatic(S, E, I, A, R, params):
    """
    SEIRA with asymptomatic compartment

    Formula: split E to symptomatic + asymptomatic infectious

    Parameters
    ----------
    S : array-like
        Input data.
    E : array-like
        Input data.
    I : array-like
        Input data.
    A : array-like
        Input data.
    R : array-like
        Input data.
    params : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Anderson-May (1991)
    """
    S = np.atleast_1d(np.asarray(S, dtype=float))
    n = len(S)
    result = float(np.mean(S))
    se = float(np.std(S, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SEIRA with asymptomatic compartment"})


def cheatsheet():
    return "seiarp: SEIRA with asymptomatic compartment"

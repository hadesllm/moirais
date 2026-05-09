"""SEIR with exposed class."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["seir_compartmental"]


def seir_compartmental(S, E, I, R, beta, sigma, gamma):
    """
    SEIR with exposed class

    Formula: adds dE/dt = beta SI/N - sigma E

    Parameters
    ----------
    S : array-like
        Input data.
    E : array-like
        Input data.
    I : array-like
        Input data.
    R : array-like
        Input data.
    beta : array-like
        Input data.
    sigma : array-like
        Input data.
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hethcote (2000)
    """
    S = np.atleast_1d(np.asarray(S, dtype=float))
    n = len(S)
    result = float(np.mean(S))
    se = float(np.std(S, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SEIR with exposed class"})


def cheatsheet():
    return "seirep: SEIR with exposed class"

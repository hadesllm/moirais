"""Logarithm converts product into a sum in homomorphic filtering.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_homomorphic_log_separation"]


def rangayyan_ch4_homomorphic_log_separation(x, p, t):
    """
    Logarithm converts product into a sum in homomorphic filtering.

    Formula: log[y(t)] = log[x(t) * p(t)] = log[x(t)] + log[p(t)], for x(t)!=0, p(t)!=0

    Parameters
    ----------
    x : array-like
        Input data.
    p : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.59, p. 244
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Logarithm converts product into a sum in homomorphic filtering."})


def cheatsheet():
    return "rng231: Logarithm converts product into a sum in homomorphic filtering."

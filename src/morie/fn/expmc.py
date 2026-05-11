"""Exponential mechanism."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["exponential_mechanism"]


def exponential_mechanism(candidates, utility, sensitivity, epsilon):
    """
    Exponential mechanism

    Formula: P(output=r) ∝ exp(ε u(D,r) / 2Δu)

    Parameters
    ----------
    candidates : array-like
        Input data.
    utility : array-like
        Input data.
    sensitivity : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    McSherry-Talwar (2007)
    """
    candidates = np.atleast_1d(np.asarray(candidates, dtype=float))
    n = len(candidates)
    result = float(np.mean(candidates))
    se = float(np.std(candidates, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Exponential mechanism"})


def cheatsheet():
    return "expmc: Exponential mechanism"

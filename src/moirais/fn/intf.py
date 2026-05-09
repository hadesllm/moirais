"""Functional integration."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["integrate_function"]


def integrate_function(coef, basis):
    """
    Functional integration

    Formula: ∫ f(t) dt over basis

    Parameters
    ----------
    coef : array-like
        Input data.
    basis : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ramsay-Silverman (2005)
    """
    coef = np.atleast_1d(np.asarray(coef, dtype=float))
    n = len(coef)
    result = float(np.mean(coef))
    se = float(np.std(coef, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Functional integration"})


def cheatsheet():
    return "intf: Functional integration"

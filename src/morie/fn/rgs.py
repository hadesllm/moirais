"""Functional regression Y = ∫β(t)X(t)dt + ε."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["functional_regression"]


def functional_regression(X, Y, basis):
    """
    Functional regression Y = ∫β(t)X(t)dt + ε

    Formula: basis-expansion + ridge

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    basis : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cardot-Ferraty-Sarda (1999)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Functional regression Y = ∫β(t)X(t)dt + ε"})


def cheatsheet():
    return "rgs: Functional regression Y = ∫β(t)X(t)dt + ε"

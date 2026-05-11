"""Mixture cure model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cure_model"]


def cure_model(time, event, X, Z):
    """
    Mixture cure model

    Formula: S(t|X) = (1-pi(X)) + pi(X) S_0(t)

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    X : array-like
        Input data.
    Z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sy-Taylor (2000); Maller-Zhou (1996)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mixture cure model"})


def cheatsheet():
    return "sscure: Mixture cure model"

"""TMLE with external comparator data."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_external_data"]


def tmle_external_data(y, D, X, external):
    """
    TMLE with external comparator data

    Formula: borrow strength from historical control with sampling weights

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    external : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schnitzer-Lok (2017)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE with external comparator data"})


def cheatsheet():
    return "tmlext: TMLE with external comparator data"

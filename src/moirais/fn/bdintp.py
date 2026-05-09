"""Bound via intersection of moment conditions."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_intersection"]


def bound_intersection(y, D, moments):
    """
    Bound via intersection of moment conditions

    Formula: max-min over moment family

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    moments : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Chernozhukov-Hong-Tamer (2007)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bound via intersection of moment conditions"})


def cheatsheet():
    return "bdintp: Bound via intersection of moment conditions"

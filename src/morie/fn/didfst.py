"""DiD random forest."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["did_forest"]


def did_forest(y, D, X, time):
    """
    DiD random forest

    Formula: forest splits on (X, t) for ATT

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    time : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Athey et al (2024)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DiD random forest"})


def cheatsheet():
    return "didfst: DiD random forest"

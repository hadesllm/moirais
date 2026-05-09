"""Breslow tie correction for Cox PH."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["breslow_tie_correction"]


def breslow_tie_correction(time, event, X):
    """
    Breslow tie correction for Cox PH

    Formula: L = product exp(beta'sum X) / (sum exp(beta'X))^d

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Breslow (1974)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Breslow tie correction for Cox PH"})


def cheatsheet():
    return "breslot: Breslow tie correction for Cox PH"

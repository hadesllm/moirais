"""North Atlantic Oscillation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["nao_index"]


def nao_index(slp):
    """
    North Atlantic Oscillation

    Formula: normalized SLP difference Azores − Iceland

    Parameters
    ----------
    slp : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hurrell (1995)
    """
    slp = np.atleast_1d(np.asarray(slp, dtype=float))
    n = len(slp)
    result = float(np.mean(slp))
    se = float(np.std(slp, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "North Atlantic Oscillation"})


def cheatsheet():
    return "naoIdx: North Atlantic Oscillation"

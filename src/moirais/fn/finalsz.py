"""Final epidemic size (Kermack-McKendrick)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["final_epidemic_size"]


def final_epidemic_size(R0, s0):
    """
    Final epidemic size (Kermack-McKendrick)

    Formula: 1 - s_inf = R0 (s0 - s_inf)

    Parameters
    ----------
    R0 : array-like
        Input data.
    s0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kermack-McKendrick (1927); Ma-Earn (2006)
    """
    R0 = np.atleast_1d(np.asarray(R0, dtype=float))
    n = len(R0)
    result = float(np.mean(R0))
    se = float(np.std(R0, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Final epidemic size (Kermack-McKendrick)"})


def cheatsheet():
    return "finalsz: Final epidemic size (Kermack-McKendrick)"

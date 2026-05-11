"""OLR parameterization."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["outgoing_longwave"]


def outgoing_longwave(T_s):
    """
    OLR parameterization

    Formula: OLR = A + B T_s

    Parameters
    ----------
    T_s : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Budyko (1969)
    """
    T_s = np.atleast_1d(np.asarray(T_s, dtype=float))
    n = len(T_s)
    result = float(np.mean(T_s))
    se = float(np.std(T_s, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "OLR parameterization"})


def cheatsheet():
    return "olrFn: OLR parameterization"

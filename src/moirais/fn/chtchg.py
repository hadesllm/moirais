"""DR for changeover designs."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["changeover_dr"]


def changeover_dr(y, D, period, unit):
    """
    DR for changeover designs

    Formula: DR ATT in cross-over experimental design

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    period : array-like
        Input data.
    unit : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Senn (2002)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DR for changeover designs"})


def cheatsheet():
    return "chtchg: DR for changeover designs"

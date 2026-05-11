"""Critical vaccination threshold for herd immunity."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vaccination_threshold"]


def vaccination_threshold(R0):
    """
    Critical vaccination threshold for herd immunity

    Formula: p_c = 1 - 1/R0

    Parameters
    ----------
    R0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Anderson-May (1991)
    """
    R0 = np.atleast_1d(np.asarray(R0, dtype=float))
    n = len(R0)
    result = float(np.mean(R0))
    se = float(np.std(R0, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Critical vaccination threshold for herd immunity"})


def cheatsheet():
    return "vacthr: Critical vaccination threshold for herd immunity"

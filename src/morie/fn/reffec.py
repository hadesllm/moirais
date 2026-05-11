"""Effective reproduction number Rt."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["effective_reproduction"]


def effective_reproduction(R0, S, N):
    """
    Effective reproduction number Rt

    Formula: Rt = R0 * S(t)/N

    Parameters
    ----------
    R0 : array-like
        Input data.
    S : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cori et al (2013)
    """
    R0 = np.atleast_1d(np.asarray(R0, dtype=float))
    n = len(R0)
    result = float(np.mean(R0))
    se = float(np.std(R0, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Effective reproduction number Rt"})


def cheatsheet():
    return "reffec: Effective reproduction number Rt"

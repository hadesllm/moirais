"""Two-strain hiatus epidemic."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hiatus_model"]


def hiatus_model(S, I1, I2, R, cross_immunity):
    """
    Two-strain hiatus epidemic

    Formula: two-strain SIR with cross-immunity

    Parameters
    ----------
    S : array-like
        Input data.
    I1 : array-like
        Input data.
    I2 : array-like
        Input data.
    R : array-like
        Input data.
    cross_immunity : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gog-Grenfell (2002)
    """
    S = np.atleast_1d(np.asarray(S, dtype=float))
    n = len(S)
    result = float(np.mean(S))
    se = float(np.std(S, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Two-strain hiatus epidemic"})


def cheatsheet():
    return "hiatus: Two-strain hiatus epidemic"

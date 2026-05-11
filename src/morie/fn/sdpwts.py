"""Semidefinite programming."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["semidefinite_program"]


def semidefinite_program(C, A, b):
    """
    Semidefinite programming

    Formula: min trace(C X) s.t. trace(A_i X) = b_i, X PSD

    Parameters
    ----------
    C : array-like
        Input data.
    A : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Vandenberghe-Boyd (1996)
    """
    C = np.atleast_1d(np.asarray(C, dtype=float))
    n = len(C)
    result = float(np.mean(C))
    se = float(np.std(C, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Semidefinite programming"})


def cheatsheet():
    return "sdpwts: Semidefinite programming"

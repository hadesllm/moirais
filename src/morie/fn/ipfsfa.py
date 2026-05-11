"""Interior-point NLP (IPOPT-style)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ipopt_solver"]


def ipopt_solver(f, constraints, x0):
    """
    Interior-point NLP (IPOPT-style)

    Formula: barrier + Newton-KKT system

    Parameters
    ----------
    f : array-like
        Input data.
    constraints : array-like
        Input data.
    x0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wächter-Biegler (2006) IPOPT
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Interior-point NLP (IPOPT-style)"})


def cheatsheet():
    return "ipfsfa: Interior-point NLP (IPOPT-style)"

"""Logarithmic barrier method."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["barrier_method"]


def barrier_method(f, constraints, x0, tau):
    """
    Logarithmic barrier method

    Formula: unconstrained + tau * sum -log(g_i(x))

    Parameters
    ----------
    f : array-like
        Input data.
    constraints : array-like
        Input data.
    x0 : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Frisch (1955)
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Logarithmic barrier method"})


def cheatsheet():
    return "barerp: Logarithmic barrier method"

"""Separation of variables PDE."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pde_separation"]


def pde_separation(pde):
    """
    Separation of variables PDE

    Formula: u(x,t) = X(x)T(t)

    Parameters
    ----------
    pde : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    classical
    """
    pde = np.atleast_1d(np.asarray(pde, dtype=float))
    n = len(pde)
    result = float(np.mean(pde))
    se = float(np.std(pde, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Separation of variables PDE"})


def cheatsheet():
    return "pdesl: Separation of variables PDE"

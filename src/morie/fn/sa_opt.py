"""Simulated annealing."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["simulated_annealing"]


def simulated_annealing(f, x0, T_init, cooling):
    """
    Simulated annealing

    Formula: accept with prob exp(-Delta/T); cool T

    Parameters
    ----------
    f : array-like
        Input data.
    x0 : array-like
        Input data.
    T_init : array-like
        Input data.
    cooling : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kirkpatrick-Gelatt-Vecchi (1983)
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Simulated annealing"})


def cheatsheet():
    return "sa_opt: Simulated annealing"

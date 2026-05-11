"""Naive gross treatment-effect bound."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_naive_gross"]


def bound_naive_gross(y, D):
    """
    Naive gross treatment-effect bound

    Formula: [E[Y|D=1] - max Y, E[Y|D=1] - min Y]

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Manski (1990)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Naive gross treatment-effect bound"})


def cheatsheet():
    return "bndnvg: Naive gross treatment-effect bound"

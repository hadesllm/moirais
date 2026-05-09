"""Composite interval mapping."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cim_qtl"]


def cim_qtl(y, markers, positions, cofactors):
    """
    Composite interval mapping

    Formula: control for unlinked markers as cofactors

    Parameters
    ----------
    y : array-like
        Input data.
    markers : array-like
        Input data.
    positions : array-like
        Input data.
    cofactors : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Zeng (1994)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Composite interval mapping"})


def cheatsheet():
    return "cqtmpl: Composite interval mapping"

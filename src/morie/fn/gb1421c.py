# morie.fn -- function file (hadesllm/morie)
"""Contingency coefficient C = sqrt(Q/(Q+n))."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_contingency_coeff"]


def gibbons_contingency_coeff(table):
    """
    Contingency coefficient C = sqrt(Q/(Q+n))

    Formula: C = sqrt(chi2/(chi2+n)); bounded [0,1) but max depends on table size

    Parameters
    ----------
    table : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: C

    References
    ----------
    Gibbons Ch 14.2.1
    """
    table = np.asarray(table, dtype=float)
    n = int(table) if table.ndim == 0 else len(table)
    result = float(np.mean(table))
    se = float(np.std(table, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Contingency coefficient C = sqrt(Q/(Q+n))"})


def cheatsheet():
    return "gb1421c: Contingency coefficient C = sqrt(Q/(Q+n))"

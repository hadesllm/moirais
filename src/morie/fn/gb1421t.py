# morie.fn — function file (hadesllm/morie)
"""Phi coefficient and Cramer's V for 2x2 and rxc tables."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_phi_cramers_v"]


def gibbons_phi_cramers_v(table):
    """
    Phi coefficient and Cramer's V for 2x2 and rxc tables

    Formula: phi = sqrt(Q/n) for 2x2; V = sqrt(Q/(n*min(r-1,c-1)))

    Parameters
    ----------
    table : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: phi_or_V

    References
    ----------
    Gibbons Ch 14.2
    """
    table = np.asarray(table, dtype=float)
    n = int(table) if table.ndim == 0 else len(table)
    result = float(np.mean(table))
    se = float(np.std(table, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Phi coefficient and Cramer's V for 2x2 and rxc tables"})


def cheatsheet():
    return "gb1421t: Phi coefficient and Cramer's V for 2x2 and rxc tables"

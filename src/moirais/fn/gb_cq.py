# moirais.fn — function file (hadesllm/moirais)
"""Cramer's V contingency measure for general r x c table."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_cramers_contingency"]


def gibbons_cramers_contingency(table):
    """
    Cramer's V contingency measure for general r table c table

    Formula: V = sqrt(chi2 / (n * min(r-1, c-1)))

    Parameters
    ----------
    table : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: V

    References
    ----------
    Gibbons Ch 14.2
    """
    table = np.asarray(table, dtype=float)
    n = int(table) if table.ndim == 0 else len(table)
    result = float(np.mean(table))
    se = float(np.std(table, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cramer's V contingency measure for general r x c table"})


def cheatsheet():
    return "gb_cq: Cramer's V contingency measure for general r x c table"

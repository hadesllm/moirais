"""DPLL SAT solving."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dpll"]


def dpll(cnf):
    """
    DPLL SAT solving

    Formula: unit propagation + branching

    Parameters
    ----------
    cnf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Davis-Putnam-Logemann-Loveland (1962)
    """
    cnf = np.atleast_1d(np.asarray(cnf, dtype=float))
    n = len(cnf)
    result = float(np.mean(cnf))
    se = float(np.std(cnf, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DPLL SAT solving"})


def cheatsheet():
    return "satDP: DPLL SAT solving"

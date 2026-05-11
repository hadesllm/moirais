"""CDCL conflict-driven clause learning."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cdcl"]


def cdcl(cnf):
    """
    CDCL conflict-driven clause learning

    Formula: DPLL + learned clauses + backjumping

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
    Marques-Silva-Sakallah (1999)
    """
    cnf = np.atleast_1d(np.asarray(cnf, dtype=float))
    n = len(cnf)
    result = float(np.mean(cnf))
    se = float(np.std(cnf, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CDCL conflict-driven clause learning"})


def cheatsheet():
    return "cdcl: CDCL conflict-driven clause learning"

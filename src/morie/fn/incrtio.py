"""Incidence rate ratio."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["incidence_rate_ratio"]


def incidence_rate_ratio(IR_e, IR_u):
    """
    Incidence rate ratio

    Formula: IRR = IR_exposed / IR_unexposed

    Parameters
    ----------
    IR_e : array-like
        Input data.
    IR_u : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rothman (2012)
    """
    IR_e = np.atleast_1d(np.asarray(IR_e, dtype=float))
    n = len(IR_e)
    result = float(np.mean(IR_e))
    se = float(np.std(IR_e, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Incidence rate ratio"})


def cheatsheet():
    return "incrtio: Incidence rate ratio"

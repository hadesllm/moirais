"""Stable Unit Treatment Value Assumption (SUTVA): no interference + consistency."""
import numpy as np
from ._richresult import RichResult

__all__ = ["sutva_assumption"]


def sutva_assumption(Y, T):
    """
    Stable Unit Treatment Value Assumption (SUTVA): no interference + consistency

    Formula: (1) Y_i(t) does not depend on T_j (no spillovers); (2) One version of treatment

    Parameters
    ----------
    Y : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'sutva_check': 'dict'}

    References
    ----------
    Molak Ch 8
    """
    Y = np.asarray(Y, dtype=float)
    n = int(Y) if Y.ndim == 0 else len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stable Unit Treatment Value Assumption (SUTVA): no interference + consistency"})


def cheatsheet():
    return "sutva: Stable Unit Treatment Value Assumption (SUTVA): no interference + consistency"

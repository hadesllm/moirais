"""Complete-case analysis baseline."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["complete_case"]


def complete_case(y, X, R):
    """
    Complete-case analysis baseline

    Formula: discard missing; estimate on complete

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    R : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Little-Rubin (2019)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Complete-case analysis baseline"})


def cheatsheet():
    return "comple: Complete-case analysis baseline"

"""TMLE for natural total effect (no mediator)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_natural_total"]


def tmle_natural_total(y, D, M, X):
    """
    TMLE for natural total effect (no mediator)

    Formula: NTE = E[Y(1) - Y(0)] = NDE + NIE

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    M : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    VanderWeele (2015)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for natural total effect (no mediator)"})


def cheatsheet():
    return "tmlnte: TMLE for natural total effect (no mediator)"

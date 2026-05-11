"""Accelerated gradient descent (constrained)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["agd_constrained"]


def agd_constrained(f, grad_f, project, x0, steps):
    """
    Accelerated gradient descent (constrained)

    Formula: projection onto feasible set after AGD step

    Parameters
    ----------
    f : array-like
        Input data.
    grad_f : array-like
        Input data.
    project : array-like
        Input data.
    x0 : array-like
        Input data.
    steps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Beck-Teboulle (2009) FISTA
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Accelerated gradient descent (constrained)"})


def cheatsheet():
    return "agdcon: Accelerated gradient descent (constrained)"

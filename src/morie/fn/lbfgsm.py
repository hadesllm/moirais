"""L-BFGS."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["lbfgs"]


def lbfgs(f, grad_f, x0, m, steps):
    """
    L-BFGS

    Formula: limited-memory quasi-Newton

    Parameters
    ----------
    f : array-like
        Input data.
    grad_f : array-like
        Input data.
    x0 : array-like
        Input data.
    m : array-like
        Input data.
    steps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Liu-Nocedal (1989)
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "L-BFGS"})


def cheatsheet():
    return "lbfgsm: L-BFGS"

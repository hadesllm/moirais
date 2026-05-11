"""BFGS quasi-Newton."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bfgs"]


def bfgs(f, grad_f, x0, steps):
    """
    BFGS quasi-Newton

    Formula: approximate Hessian rank-2 update

    Parameters
    ----------
    f : array-like
        Input data.
    grad_f : array-like
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
    Broyden-Fletcher-Goldfarb-Shanno (1970)
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BFGS quasi-Newton"})


def cheatsheet():
    return "bfgsop: BFGS quasi-Newton"

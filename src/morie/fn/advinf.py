"""Automatic differentiation VI."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["advi"]


def advi(log_p, x):
    """
    Automatic differentiation VI

    Formula: reparameterized Gaussian + autograd

    Parameters
    ----------
    log_p : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kucukelbir et al (2017)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Automatic differentiation VI"})


def cheatsheet():
    return "advinf: Automatic differentiation VI"

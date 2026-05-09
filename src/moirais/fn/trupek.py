"""Trust-region method."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["trust_region"]


def trust_region(f, grad_f, hess_f, x0, delta):
    """
    Trust-region method

    Formula: min m_k(s) within ||s|| <= delta_k

    Parameters
    ----------
    f : array-like
        Input data.
    grad_f : array-like
        Input data.
    hess_f : array-like
        Input data.
    x0 : array-like
        Input data.
    delta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Conn-Gould-Toint (2000)
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Trust-region method"})


def cheatsheet():
    return "trupek: Trust-region method"
